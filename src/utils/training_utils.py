import keras
import os

def compile_and_train(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return model

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    return test_loss, test_accuracy

def save_model(model, model_dir, model_name):
    ensure_dir_exists(model_dir)
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    return model_path

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
