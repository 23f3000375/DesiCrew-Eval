import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import validators as v


def test_verhoeff_known_valid():
    assert v.verhoeff_ok("2363")          # classic Verhoeff test vector
    assert not v.verhoeff_ok("2364")


def test_sample_aadhaar_flagged_not_rejected():
    st, notes, fixed = v.validate_aadhaar("1234 5678 9012")
    assert fixed == "123456789012" and st == "warn"


def test_pan_and_confusables():
    assert v.validate_pan("BPQPD3051R")[0] == "valid"
    st, notes, fixed = v.validate_pan("BPQPD305IR")   # I read instead of 1
    assert fixed == "BPQPD3051R" and st == "warn"
    assert v.validate_pan("ABCDE1234F")[0] == "warn"   # structurally ok, 'D' holder type unusual


def test_ifsc():
    assert v.validate_ifsc("SBIN0227112")[0] == "valid"
    st, _, fixed = v.validate_ifsc("SBINO227112")       # letter O for zero
    assert fixed == "SBIN0227112" and st == "warn"
    assert v.validate_ifsc("SBIN227112")[0] == "invalid"
    assert v.ifsc_bank_consistent("SBIN0227112", "State Bank of India")


def test_dl_passport_account():
    assert v.validate_dl("MH12 2021 0001234")[0] == "valid"
    assert v.validate_passport("X1234567")[0] == "valid"
    assert v.validate_account("3100 4258 912")[2] == "31004258912"


def test_dates():
    assert v.parse_date("26/04/26").isoformat() == "2026-04-26"
    assert v.parse_date("18-12-1979").isoformat() == "1979-12-18"
    assert v.validate_date("32/13/2026")[0] == "invalid"


def test_mrz_from_sample_passport():
    line = "X1234567<7IND7912185M3001010" + "<" * 14 + "08"
    assert len(line) == 44
    r = v.mrz_checks(line)
    assert r["passport_no"] is True
    print(r)
