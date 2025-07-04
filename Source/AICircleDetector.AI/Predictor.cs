﻿using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using System.Drawing;

namespace AICircleDetector.AI
{
    public static class Predictor
    {
        public static async Task<AIResult> Predict(string imagePath)
        {
            AIResult result = new();

            try
            {
                if (!Path.Exists(AIConfig.TrainingModelFullURL))
                {
                    string msg = "Predictor: Did not set the path to the Trained Model - please run Training and Validate Before!";
                    Console.WriteLine(msg);

                    return new AIResult
                    {
                        Success = false,
                        Message = msg
                    };
                }
             

                Console.WriteLine("Predictor: loading image...");
                NDArray inputTensor = LoadImage(imagePath, AIConfig.TrainerShapeSize);

                Console.WriteLine($"Input shape: {inputTensor.shape}, dtype: {inputTensor.dtype}");

                Console.WriteLine("Predictor: loading model...");
                             
                var model = keras.models.load_model(AIConfig.TrainingModelFullURL);

                model.summary();

                //model.compile(optimizer: keras.optimizers.Adam(),
                //              loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                //              metrics: new[] { "accuracy" });


                model.compile(
                    optimizer: keras.optimizers.Adam(),
                    loss: keras.losses.MeanSquaredError(), // or MeanAbsoluteError
                    metrics: new[] { "mean_absolute_error" }); // or "mse"
     

                Console.WriteLine("Predictor: running prediction...");
                var output = model.predict(inputTensor, use_multiprocessing: true,
                workers: Environment.ProcessorCount,  // Use all available threads
                max_queue_size: 32);                  // Ensures better generalization);

                //NDArray numpyArray = output.numpy();
                //Console.WriteLine($"Raw output: {numpyArray}, shape: {numpyArray.shape}");

                // Get the predicted number of circles (raw float value)
                //int predictedCircles = np.argmax(numpyArray);

                double raw = output.numpy()[0, 0];

                //double predictedCircles = raw * AIConfig.MaxCircles;

                int predictedCount = (int)Math.Round(raw);

                result.Success = true;
                result.Message = $"Predicted Number of Circles: {predictedCount}";
                Console.WriteLine(result.Message);

            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Message = $"Prediction failed: {ex.Message}";
                Console.WriteLine(result.Message);
            }

            return result;
        }

        private static NDArray LoadImage(string path, int size = 28)
        {
            using var bmp = new Bitmap(path);
            using var resized = new Bitmap(bmp, new Size(size, size));

            float[,,] data = new float[size, size, 1];

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    Color pixel = resized.GetPixel(x, y);
                    float gray = (pixel.R + pixel.G + pixel.B) / 3f / 255f; // normalize here
                    data[y, x, 0] = gray;
                }
            }

            NDArray array = np.array(data, dtype: np.float32);       // shape: [28, 28, 1]
            return np.expand_dims(array, 0);                          // shape: [1, 28, 28, 1]
        }


    }
}
