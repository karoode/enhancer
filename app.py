import os
import gc
import cv2
import json
import time
import uuid
import signal
import psutil
import shutil
import threading
import multiprocessing as mp
import numpy as np

from pathlib import Path
from flask import Flask, request, send_file, jsonify


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv(
    "GFPGAN_MODEL_PATH",
    os.path.join(BASE_DIR, "GFPGANv1.4.pth")
)

JOB_DIR = os.getenv(
    "JOB_DIR",
    os.path.join(BASE_DIR, "jobs")
)

# يعالج كل الوجوه
ONLY_CENTER_FACE = False

# لحماية السيرفر من الصور الكبيرة جداً
# إذا الصورة أصلاً 1080x720 ما راح يصغرها
MAX_PROCESS_WIDTH = int(os.getenv("MAX_PROCESS_WIDTH", "1080"))
MAX_PROCESS_HEIGHT = int(os.getenv("MAX_PROCESS_HEIGHT", "720"))

# يرجع الناتج إلى نفس قياس الصورة الأصلية إذا صغّرناها
RESTORE_TO_ORIGINAL_SIZE = True

# بعد إلغاء الطلب الحالي، انتظر حتى يستقر الرام
CANCEL_STABILIZE_SECONDS = float(os.getenv("CANCEL_STABILIZE_SECONDS", "2"))

# تقليل ضغط المعالج
# خليها 2 مبدئياً. إذا تريد أسرع خليها 4 أو 8.
CPU_THREADS = int(os.getenv("CPU_THREADS", "2"))

# جودة JPG الناتج
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "95"))

# أقصى وقت للطلب الواحد
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "600"))


# =========================================================
# LOW CPU THREAD SETTINGS - parent process
# =========================================================

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

Path(JOB_DIR).mkdir(parents=True, exist_ok=True)

state_lock = threading.Lock()

current_job = {
    "job_id": None,
    "process": None,
    "input_path": None,
    "output_path": None,
    "status_path": None,
    "started_at": None,
}


# =========================================================
# HELPERS
# =========================================================

def safe_print(*args):
    print(*args, flush=True)


def rss_mb_current_process():
    proc = psutil.Process(os.getpid())
    return round(proc.memory_info().rss / 1024 / 1024, 2)


def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_jpg_unicode(path, img, quality=95):
    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )

    if not ok:
        raise RuntimeError("Failed to encode output image.")

    buf.tofile(path)


def resize_for_processing(img, max_w, max_h):
    if max_w <= 0 or max_h <= 0:
        return img, 1.0

    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    if scale >= 1.0:
        return img, 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def restore_output_size(output, original_w, original_h):
    h, w = output.shape[:2]

    if w == original_w and h == original_h:
        return output

    return cv2.resize(
        output,
        (original_w, original_h),
        interpolation=cv2.INTER_CUBIC
    )


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def terminate_process(proc):
    if proc is None:
        return

    if not proc.is_alive:
        return

    try:
        proc.terminate()
        proc.join(timeout=3)
    except Exception:
        pass

    if proc.is_alive():
        try:
            if hasattr(proc, "kill"):
                proc.kill()
            else:
                os.kill(proc.pid, signal.SIGTERM)
            proc.join(timeout=3)
        except Exception:
            pass


def cleanup_old_jobs(max_age_seconds=3600):
    now = time.time()

    try:
        for item in Path(JOB_DIR).glob("*"):
            try:
                age = now - item.stat().st_mtime
                if age > max_age_seconds:
                    if item.is_file():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(str(item), ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


def clear_current_job_if_matches(job_id):
    with state_lock:
        if current_job.get("job_id") == job_id:
            current_job["job_id"] = None
            current_job["process"] = None
            current_job["input_path"] = None
            current_job["output_path"] = None
            current_job["status_path"] = None
            current_job["started_at"] = None


def is_still_current(job_id):
    with state_lock:
        return current_job.get("job_id") == job_id


# =========================================================
# CHILD PROCESS ENHANCEMENT
# =========================================================

def enhance_worker(
    job_id,
    input_path,
    output_path,
    status_path,
    model_path,
    max_w,
    max_h,
    cpu_threads,
    jpeg_quality
):
    """
    مهم:
    هذا يشتغل داخل child process.
    إذا وصل طلب جديد، السيرفر يقتل هذا البروسس بالكامل.
    هذا أفضل حل حتى نكدر نلغي GFPGAN فعلياً ونحرر الرام.
    """

    try:
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)

        import torch
        from gfpgan import GFPGANer

        try:
            cv2.setNumThreads(cpu_threads)
        except Exception:
            pass

        try:
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(1)
            torch.set_grad_enabled(False)
        except Exception:
            pass

        proc = psutil.Process(os.getpid())

        def rss():
            return round(proc.memory_info().rss / 1024 / 1024, 2)

        peak_ram = {"value": rss()}
        stop_monitor = threading.Event()

        def monitor_ram():
            while not stop_monitor.is_set():
                try:
                    r = rss()
                    if r > peak_ram["value"]:
                        peak_ram["value"] = r
                except Exception:
                    pass
                time.sleep(0.02)

            try:
                r = rss()
                if r > peak_ram["value"]:
                    peak_ram["value"] = r
            except Exception:
                pass

        monitor_thread = threading.Thread(target=monitor_ram, daemon=True)
        monitor_thread.start()

        start_total = time.perf_counter()

        safe_print("")
        safe_print("========================================")
        safe_print("[GFPGAN CHILD] START JOB:", job_id)
        safe_print("PID:", os.getpid())
        safe_print("Model:", model_path)
        safe_print("CPU_THREADS:", cpu_threads)
        safe_print("ONLY_CENTER_FACE:", ONLY_CENTER_FACE)
        safe_print("MAX_PROCESS_SIZE:", max_w, "x", max_h)
        safe_print("RAM start:", rss(), "MB")
        safe_print("========================================")

        model_start = time.perf_counter()

        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None
        )

        model_load_time = time.perf_counter() - model_start
        ram_after_model = rss()

        img_original = read_image_unicode(input_path)
        if img_original is None:
            raise RuntimeError("Failed to read input image.")

        original_h, original_w = img_original.shape[:2]

        img_process, resize_scale = resize_for_processing(
            img_original,
            max_w,
            max_h
        )

        process_h, process_w = img_process.shape[:2]
        ram_after_decode = rss()

        enhance_start = time.perf_counter()

        with torch.no_grad():
            cropped_faces, restored_faces, output = restorer.enhance(
                img_process,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

        enhance_time = time.perf_counter() - enhance_start

        if output is None:
            raise RuntimeError("GFPGAN output is None.")

        detected_faces = len(restored_faces) if restored_faces is not None else 0

        if RESTORE_TO_ORIGINAL_SIZE:
            output = restore_output_size(output, original_w, original_h)

        ram_after_enhance = rss()

        write_jpg_unicode(output_path, output, jpeg_quality)

        total_time = time.perf_counter() - start_total

        stop_monitor.set()
        monitor_thread.join(timeout=2)

        status = {
            "status": "ok",
            "job_id": job_id,
            "pid": os.getpid(),
            "input_path": input_path,
            "output_path": output_path,
            "model_path": model_path,
            "only_center_face": False,
            "detected_faces": int(detected_faces),
            "original_shape": [int(original_h), int(original_w), 3],
            "processing_shape": [int(process_h), int(process_w), 3],
            "output_shape": [int(output.shape[0]), int(output.shape[1]), int(output.shape[2])],
            "resize_scale": round(float(resize_scale), 4),
            "ram_after_model_load_mb": ram_after_model,
            "ram_after_decode_mb": ram_after_decode,
            "ram_after_enhance_mb": ram_after_enhance,
            "peak_ram_mb": peak_ram["value"],
            "model_load_time_sec": round(model_load_time, 3),
            "enhance_time_sec": round(enhance_time, 3),
            "total_time_sec": round(total_time, 3),
            "cpu_threads": cpu_threads,
            "jpeg_quality": jpeg_quality,
        }

        save_json(status_path, status)

        safe_print("[GFPGAN CHILD] DONE JOB:", job_id)
        safe_print("Detected faces:", detected_faces)
        safe_print("Original shape:", status["original_shape"])
        safe_print("Processing shape:", status["processing_shape"])
        safe_print("Peak RAM:", status["peak_ram_mb"], "MB")
        safe_print("Enhance time:", status["enhance_time_sec"], "sec")
        safe_print("Total time:", status["total_time_sec"], "sec")
        safe_print("Output:", output_path)

        try:
            del img_original
            del img_process
            del output
            del cropped_faces
            del restored_faces
            del restorer
        except Exception:
            pass

        gc.collect()

    except Exception as e:
        try:
            save_json(status_path, {
                "status": "error",
                "job_id": job_id,
                "error": str(e),
                "pid": os.getpid(),
            })
        except Exception:
            pass

        safe_print("[GFPGAN CHILD] ERROR JOB:", job_id, str(e))

    finally:
        try:
            stop_monitor.set()
        except Exception:
            pass

        gc.collect()


# =========================================================
# API ROUTES
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "ok": True,
        "service": "GFPGAN v1.4 face enhancer",
        "endpoints": {
            "POST /enhance": "multipart form-data image=@file",
            "GET /status": "current job status",
            "POST /cancel": "cancel current job"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "model_exists": os.path.isfile(MODEL_PATH),
        "model_path": MODEL_PATH,
        "parent_ram_mb": rss_mb_current_process(),
    })


@app.route("/status", methods=["GET"])
def status():
    with state_lock:
        proc = current_job.get("process")
        alive = bool(proc is not None and proc.is_alive())

        return jsonify({
            "ok": True,
            "current_job_id": current_job.get("job_id"),
            "alive": alive,
            "pid": proc.pid if proc is not None else None,
            "started_at": current_job.get("started_at"),
            "parent_ram_mb": rss_mb_current_process(),
        })


@app.route("/cancel", methods=["POST"])
def cancel():
    with state_lock:
        proc = current_job.get("process")
        job_id = current_job.get("job_id")

        if proc is not None and proc.is_alive():
            safe_print("[SERVER] Manual cancel job:", job_id)
            terminate_process(proc)

        current_job["job_id"] = None
        current_job["process"] = None
        current_job["input_path"] = None
        current_job["output_path"] = None
        current_job["status_path"] = None
        current_job["started_at"] = None

    gc.collect()
    time.sleep(CANCEL_STABILIZE_SECONDS)

    return jsonify({
        "ok": True,
        "cancelled_job_id": job_id,
        "parent_ram_mb": rss_mb_current_process(),
    })


@app.route("/enhance", methods=["POST"])
def enhance():
    cleanup_old_jobs()

    if not os.path.isfile(MODEL_PATH):
        return jsonify({
            "ok": False,
            "error": "GFPGAN model not found",
            "model_path": MODEL_PATH
        }), 500

    if "image" not in request.files:
        return jsonify({
            "ok": False,
            "error": "Missing image file. Use multipart form-data field name: image"
        }), 400

    uploaded = request.files["image"]

    if uploaded.filename == "":
        return jsonify({
            "ok": False,
            "error": "Empty filename"
        }), 400

    job_id = uuid.uuid4().hex

    input_path = os.path.join(JOB_DIR, f"{job_id}_input")
    output_path = os.path.join(JOB_DIR, f"{job_id}_enhanced.jpg")
    status_path = os.path.join(JOB_DIR, f"{job_id}_status.json")

    uploaded.save(input_path)

    safe_print("")
    safe_print("========================================")
    safe_print("[SERVER] NEW REQUEST:", job_id)
    safe_print("Input:", input_path)
    safe_print("Parent RAM:", rss_mb_current_process(), "MB")
    safe_print("========================================")

    # Cancel old job and start new job
    with state_lock:
        old_proc = current_job.get("process")
        old_job_id = current_job.get("job_id")

        if old_proc is not None and old_proc.is_alive():
            safe_print("[SERVER] Cancelling old job:", old_job_id)
            terminate_process(old_proc)

            safe_print("[SERVER] Waiting", CANCEL_STABILIZE_SECONDS, "seconds for RAM to stabilize...")
            gc.collect()
            time.sleep(CANCEL_STABILIZE_SECONDS)

        proc = mp.Process(
            target=enhance_worker,
            args=(
                job_id,
                input_path,
                output_path,
                status_path,
                MODEL_PATH,
                MAX_PROCESS_WIDTH,
                MAX_PROCESS_HEIGHT,
                CPU_THREADS,
                JPEG_QUALITY
            ),
            daemon=False
        )

        proc.start()

        current_job["job_id"] = job_id
        current_job["process"] = proc
        current_job["input_path"] = input_path
        current_job["output_path"] = output_path
        current_job["status_path"] = status_path
        current_job["started_at"] = time.time()

        safe_print("[SERVER] Started job:", job_id, "PID:", proc.pid)

    wait_start = time.time()

    while True:
        if not is_still_current(job_id):
            safe_print("[SERVER] Job cancelled by newer request:", job_id)
            return jsonify({
                "ok": False,
                "status": "cancelled",
                "job_id": job_id,
                "message": "This job was cancelled because a newer request arrived."
            }), 409

        if time.time() - wait_start > JOB_TIMEOUT_SECONDS:
            with state_lock:
                proc = current_job.get("process")
                if current_job.get("job_id") == job_id and proc is not None and proc.is_alive():
                    terminate_process(proc)

            clear_current_job_if_matches(job_id)

            return jsonify({
                "ok": False,
                "status": "timeout",
                "job_id": job_id,
                "timeout_seconds": JOB_TIMEOUT_SECONDS
            }), 504

        with state_lock:
            proc = current_job.get("process")

        if proc is None:
            return jsonify({
                "ok": False,
                "status": "error",
                "job_id": job_id,
                "error": "Process disappeared"
            }), 500

        if not proc.is_alive():
            proc.join(timeout=2)
            break

        time.sleep(0.1)

    # child process finished
    if not os.path.isfile(status_path):
        clear_current_job_if_matches(job_id)
        return jsonify({
            "ok": False,
            "status": "error",
            "job_id": job_id,
            "error": "Missing status file from worker"
        }), 500

    with open(status_path, "r", encoding="utf-8") as f:
        worker_status = json.load(f)

    if worker_status.get("status") != "ok":
        clear_current_job_if_matches(job_id)
        return jsonify({
            "ok": False,
            "status": "error",
            "job_id": job_id,
            "worker_status": worker_status
        }), 500

    if not os.path.isfile(output_path):
        clear_current_job_if_matches(job_id)
        return jsonify({
            "ok": False,
            "status": "error",
            "job_id": job_id,
            "error": "Output image not found"
        }), 500

    safe_print("[SERVER] Returning result for job:", job_id)
    safe_print("[SERVER] Worker metrics:", worker_status)
    safe_print("[SERVER] Parent RAM:", rss_mb_current_process(), "MB")

    clear_current_job_if_matches(job_id)

    response = send_file(
        output_path,
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="enhanced.jpg"
    )

    response.headers["X-Job-Id"] = job_id
    response.headers["X-Detected-Faces"] = str(worker_status.get("detected_faces", ""))
    response.headers["X-Peak-RAM-MB"] = str(worker_status.get("peak_ram_mb", ""))
    response.headers["X-Enhance-Time-Sec"] = str(worker_status.get("enhance_time_sec", ""))
    response.headers["X-Total-Time-Sec"] = str(worker_status.get("total_time_sec", ""))

    return response


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    safe_print("========================================")
    safe_print("GFPGAN v1.4 Enhancer Server")
    safe_print("MODEL_PATH:", MODEL_PATH)
    safe_print("JOB_DIR:", JOB_DIR)
    safe_print("ONLY_CENTER_FACE:", ONLY_CENTER_FACE)
    safe_print("MAX_PROCESS_SIZE:", MAX_PROCESS_WIDTH, "x", MAX_PROCESS_HEIGHT)
    safe_print("CPU_THREADS:", CPU_THREADS)
    safe_print("CANCEL_STABILIZE_SECONDS:", CANCEL_STABILIZE_SECONDS)
    safe_print("========================================")

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        threaded=True,
        debug=False,
        use_reloader=False
    )