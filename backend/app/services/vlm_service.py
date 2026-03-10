"""VLM (Vision-Language Model) service for defect explanation via Ollama/LLaVA."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VLMService:
    """Generates natural-language defect explanations using an Ollama-hosted VLM."""

    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_url = api_url or settings.VLM_API_URL
        self.model = model or settings.VLM_MODEL
        self.timeout = timeout

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read an image file and return its base64-encoded string."""
        raw = Path(image_path).read_bytes()
        return base64.b64encode(raw).decode("utf-8")

    async def _call_vlm(self, image_path: str, prompt: str) -> str:
        """Send a prompt + image to the VLM and return the response text."""
        if not Path(image_path).exists():
            logger.error("Image not found at %s – skipping VLM call.", image_path)
            return ""

        image_b64 = self._encode_image(image_path)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
        except httpx.TimeoutException:
            logger.warning("VLM request timed out after %.1fs", self.timeout)
            return ""
        except httpx.HTTPStatusError as exc:
            logger.error("VLM HTTP error %s: %s", exc.response.status_code, exc.response.text[:200])
            return ""
        except Exception:
            logger.exception("Unexpected error calling VLM service")
            return ""

    async def explain_defect(
        self,
        image_path: str,
        defect_type: str,
        extra_context: str = "",
    ) -> str:
        """Ask the VLM to describe a detected defect."""
        if not settings.VLM_ENABLED:
            return ""

        prompt = (
            f"You are an industrial quality inspection AI assistant. "
            f"A defect of type '{defect_type}' has been detected on a manufactured part. "
            f"Analyze the image and provide a concise, technical explanation of:\n"
            f"1. What the defect looks like\n"
            f"2. Likely root cause\n"
            f"3. Recommended corrective action\n"
        )
        if extra_context:
            prompt += f"\nAdditional context: {extra_context}\n"

        explanation = await self._call_vlm(image_path, prompt)
        logger.info(
            "VLM explanation generated for defect '%s' (%d chars)",
            defect_type,
            len(explanation),
        )
        return explanation

    async def detect_defects(self, image_path: str) -> dict[str, str]:
        """Use the VLM to detect defects in an image (no YOLO).

        Returns:
            dict with "verdict" ("PASS" or "FAIL") and "description".
        """
        if not settings.VLM_ENABLED:
            return {"verdict": "PASS", "description": "VLM is disabled."}

        prompt = (
            "You are a quality inspection AI. Analyze this image for any defects, "
            "damage, scratches, cracks, corrosion, discoloration, or other quality issues.\n"
            "Respond with exactly one line starting with PASS or FAIL, followed by a colon and a brief explanation.\n"
            "Examples:\n"
            'FAIL: Visible scratch on the upper surface with minor corrosion\n'
            'PASS: No visible defects, surface appears clean and uniform\n'
        )

        raw = await self._call_vlm(image_path, prompt)
        if not raw:
            return {"verdict": "PASS", "description": "VLM did not respond."}

        upper = raw.strip().upper()
        if upper.startswith("FAIL"):
            verdict = "FAIL"
        else:
            verdict = "PASS"

        # Strip the PASS:/FAIL: prefix if present
        description = raw.strip()
        for prefix in ("FAIL:", "PASS:", "FAIL :", "PASS :"):
            if description.upper().startswith(prefix):
                description = description[len(prefix):].strip()
                break

        logger.info("VLM defect detection: verdict=%s", verdict)
        return {"verdict": verdict, "description": description}

    async def ask(self, image_path: str, user_prompt: str) -> str:
        """Answer a free-form user prompt about an image."""
        if not settings.VLM_ENABLED:
            return "VLM is disabled."

        result = await self._call_vlm(image_path, user_prompt)
        logger.info("VLM prompt response (%d chars)", len(result))
        return result

    async def health_check(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            base_url = self.api_url.rsplit("/api", 1)[0]
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(base_url)
                return resp.status_code == 200
        except Exception:
            return False
