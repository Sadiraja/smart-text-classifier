from src.preprocess import clean_text


def test_lowercase():
    assert clean_text("HELLO World") == "hello world"


def test_removes_url():
    assert "http" not in clean_text("see https://example.com now")


def test_removes_email():
    assert "@" not in clean_text("mail me at a@b.com please")


def test_collapses_spaces():
    assert clean_text("a    b \n c") == "a b c"


def test_keeps_question_mark():
    assert "?" in clean_text("What time do you open?")


def test_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
