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
                Console.WriteLine("Predictor: loading image...");
                NDArray inputTensor = LoadImage(imagePath, AIConfig.ImageSize);

                Console.WriteLine($"Input shape: {inputTensor.shape}, dtype: {inputTensor.dtype}");

                Console.WriteLine("Predictor: loading model...");
                var keras = Tensorflow.KerasApi.keras;


                var model = keras.models.load_model(AIConfig.TensorFlowModel);

                Console.WriteLine("Predictor: running prediction...");
                var output = model.predict(inputTensor);

                NDArray numpyArray = output.numpy();
                Console.WriteLine($"Raw output: {numpyArray}, shape: {numpyArray.shape}");

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

        private static NDArray LoadImage(string path, int size)
        {
            using Bitmap bmp = new Bitmap(path);
            using Bitmap resized = new Bitmap(bmp, new Size(size, size));

            float[,,] data = new float[size, size, 1];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    Color pixel = resized.GetPixel(x, y);
                    float gray = (pixel.R + pixel.G + pixel.B) / 3f / 255f;
                    data[y, x, 0] = gray;
                }

            NDArray array = np.array(data, dtype: np.float32);
            return np.expand_dims(array, 0); // Shape: [1, H, W, 1]
        }
    }
}
