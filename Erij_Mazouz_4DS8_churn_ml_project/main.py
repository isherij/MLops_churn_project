import argparse
from model_pipeline import prepare_data
from model_pipeline import train_model
from model_pipeline import save_model
from model_pipeline import load_model
from model_pipeline import evaluate_model
from model_pipeline import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Predict the model")

    args = parser.parse_args()

    if args.prepare:
        x_train, x_test, y_train, y_test = prepare_data()
        print("Data prepared.")

    if args.train:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)

    if args.predict:
        x_train, x_test, y_train, y_test = prepare_data()  # Prepare les données
        model = load_model()  # Charge le modèle sauvegardé
        predict(x_test)


if __name__ == "__main__":
    main()
