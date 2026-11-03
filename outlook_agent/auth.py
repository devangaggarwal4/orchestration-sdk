"""Microsoft identity sign-in using the device-code flow with a persistent token cache."""

from __future__ import annotations

import msal

from .config import Settings


class AuthError(RuntimeError):
    pass


class GraphAuth:
    def __init__(self, settings: Settings):
        if not settings.client_id:
            raise AuthError(
                "OUTLOOK_CLIENT_ID is not set. Register an app in Azure (see README) "
                "or run with --demo to try the agent on a fake mailbox."
            )
        self.settings = settings
        self.cache = msal.SerializableTokenCache()
        if settings.token_cache.exists():
            self.cache.deserialize(settings.token_cache.read_text())
        self.app = msal.PublicClientApplication(
            settings.client_id,
            authority=f"https://login.microsoftonline.com/{settings.tenant}",
            token_cache=self.cache,
        )

    def _save(self) -> None:
        if self.cache.has_state_changed:
            path = self.settings.token_cache
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.cache.serialize())
            path.chmod(0o600)

    def token(self) -> str:
        """Return a valid access token, prompting for device-code sign-in if needed."""
        scopes = self.settings.scopes
        result = None
        accounts = self.app.get_accounts()
        if accounts:
            result = self.app.acquire_token_silent(scopes, account=accounts[0])
        if not result:
            flow = self.app.initiate_device_flow(scopes=scopes)
            if "user_code" not in flow:
                raise AuthError(f"Could not start device flow: {flow.get('error_description', flow)}")
            print(flow["message"], flush=True)
            result = self.app.acquire_token_by_device_flow(flow)
        self._save()
        if "access_token" not in result:
            raise AuthError(result.get("error_description", "Sign-in failed"))
        return result["access_token"]

    def sign_out(self) -> None:
        for account in self.app.get_accounts():
            self.app.remove_account(account)
        self._save()
