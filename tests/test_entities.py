"""
Tests for entity extraction.
"""
import pytest
from app.agent.entities import extract_entities


@pytest.mark.parametrize(
    "message,expected_key,expected_value",
    [
        ("Where is order #12345?", "order_id", "12345"),
        ("Track order 99999", "order_id", "99999"),
        ("My order 55555 hasn't arrived", "order_id", "55555"),
        ("Change my email to alice@example.com", "new_value", "alice@example.com"),
        ("I'm verified", "verified", True),
        ("OTP confirmed", "verified", True),
        ("verified", "verified", True),
    ],
)
def test_entity_extraction(message: str, expected_key: str, expected_value):
    entities = extract_entities(message)
    assert entities.get(expected_key) == expected_value, (
        f"Message: '{message}'\n"
        f"Expected entities[{expected_key!r}] == {expected_value!r}\n"
        f"Got: {entities}"
    )


def test_email_field_detected():
    entities = extract_entities("Please change my email to bob@test.org")
    assert entities.get("field") == "email"
    assert entities.get("new_value") == "bob@test.org"


def test_phone_field_detected():
    entities = extract_entities("Update my phone to +44 7911 123456")
    assert entities.get("field") == "phone"


def test_address_field_detected():
    entities = extract_entities("Change my address to 10 Downing Street, London")
    assert entities.get("field") == "address"


def test_no_entities():
    entities = extract_entities("hello how are you")
    assert entities == {}
