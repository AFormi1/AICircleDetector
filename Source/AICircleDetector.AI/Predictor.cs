using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using System.Drawing;

namespace AICircleDetector.AI
{
    public static class Predictor
    {
        public static string Predict(string imagePath)
        {
            try
            {
                if (!Path.Exists(AIConfig.TrainingModelFullURL))
                {
                    string msg = "Predictor: Did not set the path to the Trained Model - please run Training and Validate Before!";
                    return msg;
                }


                Console.WriteLine("Predictor: loading image...");
                NDArray inputTensor = LoadImage(imagePath, AIConfig.ImageShape);

                Console.WriteLine($"Input shape: {inputTensor.shape}, dtype: {inputTensor.dtype}");

                Console.WriteLine("Predictor: loading model...");

                var model = keras.models.load_model(AIConfig.TrainingModelFullURL);

                model.summary();

                model.compile(
                    optimizer: keras.optimizers.Adam(learning_rate: 0.001f),
                    loss: keras.losses.MeanSquaredError(),
                    metrics: new[] { "mean_absolute_error" });


                Console.WriteLine("Predictor: running prediction...");

                Tensors output = model.predict(
                    inputTensor,
                    use_multiprocessing: true,
                    workers: Environment.ProcessorCount,
                    max_queue_size: 32);


                NDArray predictions = output.numpy();

                int numBoxes = (int)predictions.shape.dims[1];
                int detectedBoxes = 0;

                string bb = "i:  [ xMin ] [ xMax ] [ yMin ] [ yMax ]" + Environment.NewLine;

                for (int i = 0; i < numBoxes; i++)
                {
                    float xMin = (float)predictions[0, i, 0];
                    float yMin = (float)predictions[0, i, 1];
                    float xMax = (float)predictions[0, i, 2];
                    float yMax = (float)predictions[0, i, 3];

                    bb += $"{i}: [{xMin:F4}] [{xMax:F4}] [{yMin:F4}] [{yMax:F4}]";

                    float width = xMax - xMin;
                    float height = yMax - yMin;
                    float ratio = width / height;
                    float area = width * height;

                    // Check coordinate validity
                    if (xMin < 0f || yMin < 0f || xMax > 1f || yMax > 1f)
                    {
                        bb += " --> " + "< 0 || > 1" + Environment.NewLine;
                        continue;
                    }
                    if (xMin >= xMax || yMin >= yMax)
                    {
                        bb += " --> " + "Min > Max" + Environment.NewLine;
                        continue;
                    }

                    // Filter out images, with are not sqared (+- 10%)                    
                    if (ratio < 0.85f || ratio > 1.15f)
                    {
                        bb += " --> " + "Ratio" + Environment.NewLine;
                        continue;
                    }

                    if (area < 0.005f)
                    {
                        bb += " --> " + "Area" + Environment.NewLine;
                        continue;
                    }

                    detectedBoxes++;

                    bb += " --> " + "OK" + Environment.NewLine;

                }

                return $"Prediction completed:\r\nNumber of detected Circles: {detectedBoxes}\r\nBoundingBoxes:\r\n{bb}";                             

            }
            catch (Exception ex)
            {
                return $"Prediction failed:\r\n{ex}";
            }

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
