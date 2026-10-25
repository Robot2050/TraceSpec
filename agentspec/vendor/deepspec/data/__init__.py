"""Vendored copy: only the chat-message encoding helpers are needed by the
evaluation path (training dataset classes are intentionally not vendored)."""

from .parser import TEMPLATE_REGISTRY, encode_chat_messages, render_chat_messages

__all__ = ["TEMPLATE_REGISTRY", "encode_chat_messages", "render_chat_messages"]
