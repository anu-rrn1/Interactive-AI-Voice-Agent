"""Pydantic response models for the HTTP API."""

from pydantic import BaseModel


class PersonaInfo(BaseModel):
    """A persona the user can pick in the UI."""

    id: str
    name: str
    description: str


class LanguageInfo(BaseModel):
    """A target language for the live translator."""

    code: str
    label: str


class PersonasResponse(BaseModel):
    """Response of ``GET /personas``."""

    personas: list[PersonaInfo]
    languages: list[LanguageInfo]
    default_persona: str
    default_language: str
