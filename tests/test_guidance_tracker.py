from guidance.tracker import _parse_promises


def test_invalid_status_becomes_pending():
    result = _parse_promises(
        '[{"category":"Demand","promise":"Growth","source_quarter":"Q1_FY27",'
        '"evaluation":"No later evidence","status":"Unknown","citation":"q1.pdf | Q1_FY27 | page 2"}]'
    )
    assert result[0].status == "Pending"
