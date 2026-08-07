from . import basic

while True:
    text = input('basic > ')
    if text.strip() == '': continue

    result, _ = basic.run_ai('<stdin>', text)

    if result.error: print(result.error.as_string())
    elif result.value is not None and result.value is not basic.Number.null:
        if len(result.value.elements) == 1:
            print(repr(result.value.elements[0]))
        else:
            print(repr(result.value))
