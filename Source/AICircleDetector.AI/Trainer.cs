﻿using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;
using System.Drawing;
using Tensorflow.Keras;
using System.Threading;
using OneOf.Types;

namespace AICircleDetector.AI
{
    public static class Trainer
    {
        public static async Task<AIResult> Train(CancellationToken cancellationToken, string basepath)
        {
            AIResult result = new();
      
            try
            {
                string csvPath = Path.Combine(basepath, "labels.csv");

                (NDArray images, NDArray labels) = LoadDataset(cancellationToken, csvPath, basepath);

                if (cancellationToken.IsCancellationRequested)
                {
                    return result;
                }

                result = TrainModel(cancellationToken, images, labels);

                if (cancellationToken.IsCancellationRequested)
                {
                    return result;
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                result.Success = false;
                result.Message = ex.Message;
            }

            return result;
        }




        public static (NDArray images, NDArray labels) LoadDataset(CancellationToken cancellationToken, string csvPath, string imagesFolder)
        {
            string[] lines = File.ReadAllLines(csvPath).Skip(1).ToArray(); // Skip header
            int count = lines.Length;

            float[,,,] imagesArray = new float[count, AIConfig.ImageSize, AIConfig.ImageSize, 1];
            float[] labelsArray = new float[count];

            for (int i = 0; i < count; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    return (0.0, 0.0);
                }

                string[] parts = lines[i].Split(';');
                string filename = parts[0];
                int label = int.Parse(parts[1]);

                if (label > AIConfig.MaxCircles)
                    throw new Exception($"Label {label} exceeds MaxCircleClasses ({AIConfig.MaxCircles}).");

                string imagePath = Path.Combine(imagesFolder, filename);
                using Bitmap bmp = new Bitmap(imagePath);
                using Bitmap resized = new Bitmap(bmp, new Size(AIConfig.ImageSize, AIConfig.ImageSize));

                for (int y = 0; y < AIConfig.ImageSize; y++)
                {
                    for (int x = 0; x < AIConfig.ImageSize; x++)
                    {
                        Color pixel = resized.GetPixel(x, y);
                        float gray = (pixel.R + pixel.G + pixel.B) / 3f / 255f;
                        imagesArray[i, y, x, 0] = gray;
                    }
                }

                labelsArray[i] = label / (float)AIConfig.MaxCircles;
            }

            Console.WriteLine("Trainer.LoadDataset() completed");

            NDArray images = np.array(imagesArray);
            NDArray labels = np.array(labelsArray).reshape(new Shape(count, 1));

            return (images, labels);
        }


        public static AIResult TrainModel(CancellationToken cancellationToken, NDArray images, NDArray labels)
        {
            var keras = Tensorflow.KerasApi.keras;
            var layers = keras.layers;

            // Model Architecture
            var input = keras.Input(shape: (AIConfig.ImageSize, AIConfig.ImageSize, 1));

            // First convolutional layer
            var x = layers.Conv2D(64, 3, activation: keras.activations.Relu).Apply(input);
            x = layers.BatchNormalization().Apply(x);
            x = layers.MaxPooling2D().Apply(x);

            // Second convolutional layer
            x = layers.Conv2D(64, 3, activation: keras.activations.Relu).Apply(x);
            x = layers.BatchNormalization().Apply(x);
            x = layers.MaxPooling2D().Apply(x);

            // Flatten the output and apply fully connected layers with regularization
            x = layers.Flatten().Apply(x);
            x = layers.Dense(256, activation: keras.activations.Relu).Apply(x);  // Increase neurons here
            x = layers.Dropout(0.5f).Apply(x);


            // Output layer (single value for regression)
            //var output = layers.Dense(1, activation: keras.activations.Linear).Apply(x);
            var output = layers.Dense(1).Apply(x);  // No activation function for regression output


            var model = keras.Model(input, output);

            //model.compile(
            //    optimizer: keras.optimizers.Adam(),
            //    loss: keras.losses.MeanSquaredError(),
            //    metrics: new[] { "mae" }
            //);

            model.compile(
                optimizer: keras.optimizers.Adam(learning_rate: 0.001f),
                loss: keras.losses.MeanAbsoluteError(),
                metrics: new[] { "mae" }  // Keep MAE as a metric
            );


            // Initialize early stopping parameters
            int patience = 15;
            int bestEpoch = 0;
            float bestValLoss = float.MaxValue;
            int epochsWithoutImprovement = 0;

            // Training loop with manual early stopping
            int totalEpochs = 50;
            for (int epoch = 0; epoch < totalEpochs; epoch++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    return new AIResult();
                }

                var history = model.fit(images, labels, batch_size: 50, epochs: 1, validation_split: 0.2f, verbose: 1);
                float valLoss = history.history["val_loss"].Last(); // Get validation loss for current epoch

                // Early stopping check
                if (valLoss < bestValLoss)
                {
                    bestValLoss = valLoss;
                    bestEpoch = epoch;
                    epochsWithoutImprovement = 0;
                }
                else
                {
                    epochsWithoutImprovement++;
                }

                if (epochsWithoutImprovement >= patience)
                {
                    Console.WriteLine($"Early stopping at epoch {epoch + 1}. No improvement in validation loss for {patience} epochs.");
                    break;
                }
            }

            // Final metrics
            // Use a variable to hold the results
            var evalResults = model.evaluate(images, labels, verbose: 0);

            evalResults.TryGetValue("loss", out float finalLoss);
            evalResults.TryGetValue("mean_absolute_error", out float finalMAE);

            // Save the model
            model.save(AIConfig.TensorFlowModel);

            return new AIResult()
            {
                Success = true,
                Message = $"Final loss: {finalLoss:F4}\nFinal MAE: {finalMAE:F4}",
                Loss = finalLoss,
                MAE = finalMAE
            };
        }



    }
}
