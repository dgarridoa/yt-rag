from __future__ import annotations

import json
import os
from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pydantic import BaseModel, SecretStr


def get_secret_client(key_vault_name: str) -> SecretClient:
    key_vault_url = f"https://{key_vault_name}.vault.azure.net"
    credential = DefaultAzureCredential(resource=key_vault_url)
    client = SecretClient(
        vault_url=key_vault_url,
        credential=credential,
    )
    return client


def get_secret_value_from_client(
    client: SecretClient, secret_name: str
) -> SecretStr:
    value = client.get_secret(secret_name).value
    if value is None:
        raise ValueError(f"Secret {secret_name} not found.")
    return SecretStr(value)


class Settings(BaseModel):
    env: str
    databricks_host: SecretStr
    databricks_token: SecretStr
    api_username: SecretStr
    api_password: SecretStr

    @classmethod
    def get_settings_from_client(cls, client: SecretClient):
        settings = cls(
            env=get_secret_value_from_client(client, "env").get_secret_value(),
            databricks_host=get_secret_value_from_client(
                client, "databricks-host"
            ),
            databricks_token=get_secret_value_from_client(
                client, "databricks-token"
            ),
            api_username=get_secret_value_from_client(client, "api-username"),
            api_password=get_secret_value_from_client(client, "api-password"),
        )
        os.environ["DATABRICKS_HOST"] = (
            settings.databricks_host.get_secret_value()
        )
        os.environ["DATABRICKS_TOKEN"] = (
            settings.databricks_token.get_secret_value()
        )
        with open("config.share", "w") as f:
            delta_sharing_credential = json.loads(
                get_secret_value_from_client(
                    client, "delta-sharing-credential"
                ).get_secret_value()
            )
            json.dump(delta_sharing_credential, f)

        return settings


@lru_cache
def get_settings():
    key_vault_name = os.getenv("KEY_VAULT_NAME", "kv-yt-rag")
    secret_client = get_secret_client(key_vault_name)
    settings = Settings.get_settings_from_client(secret_client)
    return settings
