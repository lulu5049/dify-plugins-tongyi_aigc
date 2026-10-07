# author: sawyer-shi

from typing import Any

import requests
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from tools.bailian_endpoints import compatible_base_url


class TongyiAigcProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            api_key = credentials.get("api_key")
            if not api_key:
                raise ToolProviderCredentialValidationError("Bailian API key is required")
            if len(api_key) < 10:
                raise ToolProviderCredentialValidationError("Bailian API key length is invalid")
            if not credentials.get("base_url"):
                raise ToolProviderCredentialValidationError("Bailian Base URL is required")

            compatible_base_url(credentials)
            self._test_bailian_connection(credentials)
        except ToolProviderCredentialValidationError:
            raise
        except Exception as e:
            raise ToolProviderCredentialValidationError(
                f"Bailian credential validation failed: {str(e)}"
            )

    def _test_bailian_connection(self, credentials: dict[str, Any]) -> None:
        api_key = str(credentials["api_key"])
        url = f"{compatible_base_url(credentials)}/models"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
        except requests.RequestException as req_err:
            raise ToolProviderCredentialValidationError(
                f"Unable to reach Bailian service: {req_err}"
            )

        if response.status_code != 200:
            try:
                data = response.json()
                error_message = (
                    data.get("error", {}).get("message")
                    or data.get("message")
                    or response.text
                )
            except Exception:
                error_message = response.text
            raise ToolProviderCredentialValidationError(
                f"Bailian API error {response.status_code}: {error_message}"
            )
