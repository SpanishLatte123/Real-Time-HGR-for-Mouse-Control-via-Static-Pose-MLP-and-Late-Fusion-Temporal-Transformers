from model.static_model import ClassificationModel
import os


def main():
    #params
    path_to_model = os.path.join(os.getcwd(), "trained_models\static\static_model_pass3.h5")
    path_to_data = os.path.join(os.getcwd(), "data\combined_dataset1")
    hidden_layers = (50,25,10)
    learning_rate = 0.01
    epochs = 15
    test_size = 0.3

    # Train the model
    model = ClassificationModel()
    model.read_dataset(path_to_data)
    model.process_dataset()
    model.train(hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                epochs=epochs,
                test_size=test_size)
    model.save_model(path_to_model)
    
if __name__ == '__main__':
    main()