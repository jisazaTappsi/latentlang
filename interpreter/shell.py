from . import basic

while True:
    text = input('basic > ')
    if text.strip() == '': continue

    result, _ = basic.run_ai('<stdin>', text, force_interpreter=False)

    if result.error:
        print(result.error.as_string())
    elif result.value is not None:
        value = result.value.elements[-1] if result.value.elements else None
        if value is not None and value is not basic.Number.null:
            print(repr(value))
