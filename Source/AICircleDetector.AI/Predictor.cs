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
                if (!Directory.Exists(AIConfig.TrainingModelFullURL))
                {
                    return "Predictor: Model path not set or model not trained yet.";
                }

                int maxCircles = AIConfig.MaxCircles;

                NDArray inputTensor = LoadImage(imagePath, AIConfig.ImageShape);

                var model = keras.models.load_model(AIConfig.TrainingModelFullURL);

                Tensors output = model.predict(inputTensor);

                var predictions = output.numpy();
                var batchPredictions = predictions[0]; // Erstes (und einziges) Batch

                float predCountRaw = batchPredictions[batchPredictions.size - 1];

                int circleCount = (int)Math.Round(predCountRaw * maxCircles);


                // Get bounding box values: first maxCircles * 4 entries
                var bboxFlat = batchPredictions[$"0:{maxCircles * 4}"]; // Slice string
                var boxes = bboxFlat.reshape(new Shape(maxCircles, 4));            // Shape: (maxCircles, 4)

                var sb = new System.Text.StringBuilder();
                sb.AppendLine($"Predicted Circle Count (scaled): {circleCount}");
                sb.AppendLine("Bounding Boxes:");

                for (int i = 0; i < maxCircles; i++)
                {
                    float xMin = (float)boxes[i][0];
                    float yMin = (float)boxes[i][1];
                    float xMax = (float)boxes[i][2];
                    float yMax = (float)boxes[i][3];

                    sb.AppendLine($"Box {i + 1}: [{xMin:F6}, {yMin:F6}, {xMax:F6}, {yMax:F6}]");
                }

                return sb.ToString();
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
