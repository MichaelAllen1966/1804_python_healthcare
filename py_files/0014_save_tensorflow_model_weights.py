# Save the weights
model.save_weights('filename')

# Restore the weights
model = create_model()
model.load_weights('filename')