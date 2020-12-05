from src import download_dataset, parse
from models import MLP_Simple, DeepChrome, DeepNeighbor_VAE
from src.evaluation import evaluate_model

download_dataset.check_if_dataset_exists()

X_train, X_test, Y_train, Y_test = parse.create_dataset('dataset/data/E100')


# Train a Simple MLP model
simple_model = MLP_Simple.MLP_Simple()
simple_model.compile()
simple_model.fit(X_train, Y_train)

print("MLP\n", evaluate_model(simple_model, X_test, Y_test))

# Fitting a VAE model to create a latent representation of 
vae = DeepNeighbor_VAE.VAE(_latent_dim=10)
vae.fit(X_train)

X_train_encoded = vae.generate_encoded(X_train)

X_test_encoded = vae.generate_encoded(X_test)

print(X_train_encoded.shape)

# print("X_train shape: {}".format(X_train_encoded.shape))

# Training simple model with encoded input
simple_encoded_model = MLP_Simple.MLP_Simple()
simple_encoded_model.compile()
simple_encoded_model.fit(X_train_encoded, Y_train)

print("MLP_Encoded\n", evaluate_model(simple_encoded_model, X_test_encoded, Y_test))



# # Train DeepChrome
# deepchome_model = DeepChrome.DeepChrome()
# deepchome_model.compile()
# deepchome_model.fit(X_train, Y_train)
# print("DeepChrome\n", evaluate_model(deepchome_model, X_test, Y_test))

# vae = DeepNeighbor_VAE.VAE(_latent_dim=10)
# vae.fit(X_train)



# Train Attentive Chrome
# attentive_chrome = AttentiveChrome.AttentiveChrome()
    
