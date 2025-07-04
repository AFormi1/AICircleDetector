using Tensorflow.NumPy;
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
                if (string.IsNullOrEmpty(AIConfig.TrainingModelFullURL))
                {
                    string msg = "Predictor: Did not set the path to the Trained Model - please run Training and Validate Before!";
                    result.Success = false;
                    result.Message = $"{msg}";
                    Console.WriteLine(msg);

                    return result;
                }

                Console.WriteLine("Predictor: loading image...");
                NDArray inputTensor = LoadImage(imagePath, AIConfig.TrainerShapeSize);

                Console.WriteLine($"Input shape: {inputTensor.shape}, dtype: {inputTensor.dtype}");

                Console.WriteLine("Predictor: loading model...");
                             
                var model = keras.models.load_model(AIConfig.TrainingModelFullURL);

                model.summary();

                model.compile(optimizer: keras.optimizers.Adam(),
                              loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                              metrics: new[] { "accuracy" });


                Console.WriteLine("Predictor: running prediction...");
                var output = model.predict(inputTensor);

                NDArray numpyArray = output.numpy();
                Console.WriteLine($"Raw output: {numpyArray}, shape: {numpyArray.shape}");

                // Get the predicted number of circles (raw float value)
                float predictedCircles = numpyArray[0, 0] * AIConfig.MaxCircles;

                result.Success = true;
                result.Message = $"Predicted Number of Circles: {predictedCircles:F4}";
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
