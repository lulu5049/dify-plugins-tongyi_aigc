import time
from collections.abc import Generator
from typing import Any

import requests
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.bailian_endpoints import native_base_url


TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}


def wait_for_bailian_task(
    tool: Any,
    credentials: dict[str, Any],
    task_id: str,
    *,
    poll_interval_seconds: int = 10,
    max_wait_seconds: int = 540,
    heartbeat_seconds: int = 30,
) -> Generator[ToolInvokeMessage, None, dict[str, Any] | None]:
    """Poll a Bailian async task until it reaches a terminal state.

    The generator emits periodic progress messages so Dify's streaming
    connection remains active while the tool invocation is running.
    """
    api_key = credentials.get("api_key")
    if not api_key:
        yield tool.create_text_message("❌ API密钥未配置")
        return None

    poll_interval_seconds = max(3, min(int(poll_interval_seconds), 60))
    max_wait_seconds = max(30, int(max_wait_seconds))
    heartbeat_seconds = max(poll_interval_seconds, int(heartbeat_seconds))

    api_url = f"{native_base_url(credentials)}/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    started = time.monotonic()
    last_status = ""
    last_heartbeat = -heartbeat_seconds

    while True:
        elapsed = int(time.monotonic() - started)
        if elapsed >= max_wait_seconds:
            yield tool.create_text_message(
                f"⌛ 已等待 {elapsed}s，任务仍未结束。Task ID: {task_id}。"
                "任务可能仍在百炼后台排队，可稍后用“视频结果查询”继续查询。"
            )
            return None

        try:
            response = requests.get(api_url, headers=headers, timeout=30)
        except requests.exceptions.Timeout:
            if elapsed - last_heartbeat >= heartbeat_seconds:
                yield tool.create_text_message(
                    f"⏳ 查询暂时超时，任务仍在等待中（{elapsed}s）。Task ID: {task_id}"
                )
                last_heartbeat = elapsed
            time.sleep(poll_interval_seconds)
            continue
        except requests.exceptions.RequestException as exc:
            yield tool.create_text_message(f"❌ 查询任务状态失败: {exc}")
            return None

        if response.status_code != 200:
            yield tool.create_text_message(
                f"❌ 查询任务状态失败，HTTP {response.status_code}: {response.text[:500]}"
            )
            return None

        try:
            data = response.json()
        except ValueError:
            yield tool.create_text_message(
                f"❌ 查询任务状态返回非JSON内容: {response.text[:500]}"
            )
            return None

        output = data.get("output", {}) or {}
        status = str(output.get("task_status") or "UNKNOWN").upper()

        if status != last_status:
            yield tool.create_text_message(
                f"🔄 视频任务状态: {status}（已等待 {elapsed}s）"
            )
            last_status = status
            last_heartbeat = elapsed
        elif elapsed - last_heartbeat >= heartbeat_seconds:
            yield tool.create_text_message(
                f"⏳ 视频任务仍为 {status}（已等待 {elapsed}s）"
            )
            last_heartbeat = elapsed

        if status in TERMINAL_STATUSES:
            return data

        time.sleep(poll_interval_seconds)


def emit_final_video_result(
    tool: Any,
    result_data: dict[str, Any],
) -> Generator[ToolInvokeMessage, None, None]:
    output = result_data.get("output", {}) or {}
    status = str(output.get("task_status") or "UNKNOWN").upper()
    task_id = output.get("task_id", "unknown")

    if status == "SUCCEEDED":
        video_url = output.get("video_url") or ""
        last_frame_url = output.get("last_frame_url") or ""
        yield tool.create_text_message(f"✅ 视频生成完成！Task ID: {task_id}")
        if video_url:
            yield tool.create_text_message(f"🎬 视频链接: {video_url}")
        if last_frame_url:
            yield tool.create_text_message(f"🖼️ 尾帧链接: {last_frame_url}")
        yield tool.create_json_message(result_data)
        return

    error_code = output.get("code", "Unknown")
    error_message = output.get("message", "Unknown error")
    yield tool.create_text_message(
        f"❌ 视频任务失败\n"
        f"Task ID: {task_id}\n"
        f"Status: {status}\n"
        f"Error Code: {error_code}\n"
        f"Error Message: {error_message}"
    )
    yield tool.create_json_message(result_data)
