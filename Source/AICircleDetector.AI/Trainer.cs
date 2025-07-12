using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;
using ProtoBuf;
using System.Drawing;
using System.Reflection.Metadata.Ecma335;
using OneOf.Types;
using Tensorflow.Keras.Callbacks;
using Tensorflow.Keras.Engine;
using System.Text;
using System.Threading.Channels;
using System.Threading;
using System.Linq;
using Tensorflow.Keras.Losses;
using Tensorflow.Util;

namespace AICircleDetector.AI
{
    public static class Trainer
    {

        public static string Train(string basepath)
        {
            try
            {
                int epochs = 100;
                int batchSize = 128;

                // Paths
                string modelPath = AIConfig.TrainingModelFullURL;
                string trainTfPath = Path.Combine(basepath, AIConfig.TrainingTF);
                string validationTfPath = Path.Combine(basepath, AIConfig.ValidateTF);

                // Build and compile model
                IModel model;
                if (Directory.Exists(modelPath) && Directory.GetDirectories(modelPath).Length > 0)
                {
                    model = keras.models.load_model(AIConfig.TrainingModelFullURL);
                }
                else
                {
                    Tensor input = keras.Input(shape: (AIConfig.ImageShape, AIConfig.ImageShape, 1));

                    var x = keras.layers.Conv2D(32, 3, activation: keras.activations.Relu).Apply(input);
                    x = keras.layers.MaxPooling2D().Apply(x);
                    x = keras.layers.Conv2D(64, 3, activation: keras.activations.Relu).Apply(x);
                    x = keras.layers.MaxPooling2D().Apply(x);
                    x = keras.layers.Flatten().Apply(x);
                    x = keras.layers.Dense(128, activation: keras.activations.Relu).Apply(x);

                    var bbox_output = keras.layers.Dense(AIConfig.MaxCircles * 4, activation: keras.activations.Sigmoid).Apply(x);
                    bbox_output = keras.layers.Reshape((AIConfig.MaxCircles, 4)).Apply(bbox_output);

                    model = keras.Model(input, bbox_output);
                }

                model.summary();

                model.compile(
                    optimizer: keras.optimizers.Adam(),
                    loss: keras.losses.MeanSquaredError(),
                    metrics: new[] { "mean_absolute_error" }
                );

                List<Example> trainData = LoadTFRecord(trainTfPath);
                List<Example> validationData = LoadTFRecord(validationTfPath);

                NDArray xTrain, bboxTrain, xValidate, bboxValidate;
                Create_ND_array(trainData, out xTrain, out bboxTrain);
                Create_ND_array(validationData, out xValidate, out bboxValidate);

                ValidationDataPack validationPack = new ValidationDataPack((xValidate, bboxValidate));


                string xt = $"xTrain shape: {xTrain.shape}, dtype: {xTrain.dtype}"; //expected format like: "xTrain shape: (737, 28, 28, 1), dtype: TF_FLOAT"
                string bt = $"bboxTrain shape: {bboxTrain.shape}, dtype: {bboxTrain.dtype}"; //expected format like: "bboxTrain shape: (454, 10, 4), dtype: TF_FLOAT"

                if (xTrain == null || bboxTrain == null)
                    return $"Training failed:\r\nOne of the required NDArrays is null";


                ICallback history = model.fit(
                    xTrain, bboxTrain,
                    batch_size: batchSize,
                    epochs: epochs,
                    validation_data: validationPack,
                    shuffle: true,
                    use_multiprocessing: true,
                    workers: Environment.ProcessorCount,
                    verbose: 1
                );

                // Save   
                AIConfig.TrainingModelFullURL = AIConfig.TrainingModelFullURL;
                model.save(modelPath);


                //Report
                float loss = history.history["loss"].Last();
                float val_loss = history.history["val_loss"].Last();

                float mae = history.history["mean_absolute_error"].Last();
                float val_mae = history.history["val_mean_absolute_error"].Last();

                return
                    $"Training completed:\n" +
                    $"Training Loss: {loss:F4}\n" +
                    $"Training MAE: {mae:F4}\n" +
                    $"Validation Loss: {val_loss:F4}\n" +
                    $"Validation MAE: {val_mae:F4}";

            }
            catch (Exception ex)
            {
                return $"Training failed:\r\n{ex}";
            }
        }






        private static void Create_ND_array(List<Example> trainData, out NDArray xTrain, out NDArray bboxTrain)
        {
            xTrain = null;
            bboxTrain = null;

            try
            {
                int imageChannels = 1; // z.B. Graustufen
                int imageSize = AIConfig.ImageShape * AIConfig.ImageShape;
                int maxBoxes = AIConfig.MaxCircles; // z.B. 10 Boundingboxes pro Bild

                var images = new List<float[]>();
                var allBBoxes = new List<float[,]>();

                foreach (var example in trainData)
                {
                    var featureMap = example.Features?.feature;
                    if (featureMap == null ||
                        !featureMap.TryGetValue("image/encoded", out var imageFeature) ||
                        !featureMap.TryGetValue("image/object/bbox/xmin", out var xminFeature) ||
                        !featureMap.TryGetValue("image/object/bbox/ymin", out var yminFeature) ||
                        !featureMap.TryGetValue("image/object/bbox/xmax", out var xmaxFeature) ||
                        !featureMap.TryGetValue("image/object/bbox/ymax", out var ymaxFeature))
                        continue;

                    var imageBytes = imageFeature.BytesList?.Values?.FirstOrDefault();
                    if (imageBytes == null || imageBytes.Length == 0)
                        continue;

                    var xmins = xminFeature.FloatList?.Values;
                    var ymins = yminFeature.FloatList?.Values;
                    var xmaxs = xmaxFeature.FloatList?.Values;
                    var ymaxs = ymaxFeature.FloatList?.Values;

                    if (xmins == null || ymins == null || xmaxs == null || ymaxs == null)
                        continue;

                    int boxCount = xmins.Length;
                    if (boxCount == 0 || ymins.Length != boxCount || xmaxs.Length != boxCount || ymaxs.Length != boxCount)
                        continue;

                    // Bild in Bitmap laden und auf AIConfig.ImageShape skalieren
                    using var ms = new MemoryStream(imageBytes);
                    using var bmp = new Bitmap(ms);
                    using var resized = new Bitmap(bmp, new Size(AIConfig.ImageShape, AIConfig.ImageShape));

                    // Bild in float array (Graustufen 0..1) konvertieren
                    float[] imageFloats = new float[imageSize];
                    for (int y = 0; y < AIConfig.ImageShape; y++)
                    {
                        for (int x = 0; x < AIConfig.ImageShape; x++)
                        {
                            Color pixel = resized.GetPixel(x, y);
                            float gray = (pixel.R + pixel.G + pixel.B) / 3f / 255f;
                            imageFloats[y * AIConfig.ImageShape + x] = gray;
                        }
                    }

                    images.Add(imageFloats);

                    // Boundingboxes in gepolstertes Array packen (maxBoxes x 4)
                    float[,] bboxes = new float[maxBoxes, 4];
                    int usableCount = Math.Min(boxCount, maxBoxes);

                    for (int i = 0; i < usableCount; i++)
                    {
                        bboxes[i, 0] = xmins[i];
                        bboxes[i, 1] = ymins[i];
                        bboxes[i, 2] = xmaxs[i];
                        bboxes[i, 3] = ymaxs[i];
                    }

                    allBBoxes.Add(bboxes);
                }

                if (images.Count == 0)
                    return;

                int sampleCount = images.Count;
                var reshapedImages = new float[sampleCount, AIConfig.ImageShape, AIConfig.ImageShape, imageChannels];
                var bboxArray = new float[sampleCount, maxBoxes, 4];

                // 1D Bildarrays in 4D Array (Samples, H, W, C) umwandeln
                for (int i = 0; i < sampleCount; i++)
                {
                    for (int row = 0; row < AIConfig.ImageShape; row++)
                    {
                        for (int col = 0; col < AIConfig.ImageShape; col++)
                        {
                            reshapedImages[i, row, col, 0] = images[i][row * AIConfig.ImageShape + col];
                        }
                    }

                    // Boundingboxes übertragen
                    for (int b = 0; b < maxBoxes; b++)
                    {
                        for (int k = 0; k < 4; k++)
                        {
                            bboxArray[i, b, k] = allBBoxes[i][b, k];
                        }
                    }
                }

                xTrain = np.array(reshapedImages);
                bboxTrain = np.array(bboxArray);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Create_ND_array] Error: {ex.Message}");
                xTrain = null;
                bboxTrain = null;
            }
        }






        private static List<Example> LoadTFRecord(string tfRecordPath)
        {
            List<Example> examples = new List<Example>();

            using (FileStream file = File.OpenRead(tfRecordPath))
            {
                while (file.Position < file.Length)
                {
                    // Read length (8 bytes unsigned long)
                    byte[] lengthBytes = new byte[8];
                    int readLen = file.Read(lengthBytes, 0, 8);
                    if (readLen < 8)
                        break; // EOF or corrupted

                    ulong recordLength = BitConverter.ToUInt64(lengthBytes, 0);

                    // Optional: sanity check for recordLength
                    if (recordLength > int.MaxValue)
                        throw new Exception($"Record length too large: {recordLength}");

                    // Skip length CRC (4 bytes)
                    file.Seek(4, SeekOrigin.Current);

                    // Read record data
                    byte[] data = new byte[recordLength];
                    int readData = file.Read(data, 0, (int)recordLength);
                    if (readData < (int)recordLength)
                        break; // EOF or corrupted

                    // Skip data CRC (4 bytes)
                    file.Seek(4, SeekOrigin.Current);

                    using (MemoryStream ms = new MemoryStream(data))
                    {
                        Example example = Serializer.Deserialize<Example>(ms);
                        examples.Add(example);
                    }
                }
            }

            return examples;
        }

    }
}
