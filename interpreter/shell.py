from . import basic

while True:
    text = input('basic > ')
    result, _ = basic.run_ai('<stdin>', text)
    if result.error: print(result.error.as_string())
    elif result.value: print(result.value)
