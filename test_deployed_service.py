import sys

import requests


# Domyślny adres lokalnego serwisu
DEFAULT_URL = "http://localhost:3000/predict"


# Prosty przykładowy obraz w formacie oczekiwanym przez model
# Kształt danych to [3, 28, 28]
# 3 oznacza kanały RGB, a 28x28 to rozmiar obrazu BloodMNIST
TEST_IMAGE = [
    [
        [0.0 for _ in range(28)]
        for _ in range(28)
    ]
    for _ in range(3)
]


def main():
    # Jeśli podano adres endpointu w argumencie, użyj go
    # Jeśli nie podano, użyj lokalnego adresu
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