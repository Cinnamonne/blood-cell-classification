import sys

import requests


# Domyślny adres serwera na Azure VM
DEFAULT_URL = "http://68.221.141.213:3000/predict"


# Prosty przykładowy obraz w formacie [3, 28, 28]
TEST_IMAGE = [
    [
        [0.0 for _ in range(28)]
        for _ in range(28)
    ]
    for _ in range(3)
]


def main():
    # Jeśli podano inny endpoint w argumencie, użyj go, jak nie podano, użyj domyślnego endpointu z Azure VM
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    response = requests.post(
        url,
        json={"image": TEST_IMAGE},
        timeout=30,
    )

    response.raise_for_status()

    print("Endpoint:", url)
    print("Response:")
    print(response.json())


if __name__ == "__main__":
    main()