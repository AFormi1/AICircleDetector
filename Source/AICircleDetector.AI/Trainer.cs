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

namespace AICircleDetector.AI
{
    public static class Trainer
    {

        public static string Train(string basepath)
        {
            try
            {
                int epochs = 50;

                // Paths
                string modelPath = AIConfig.TrainingModelFullURL;
                string trainTfPath = Path.Combine(basepath, AIConfig.TrainingTF);

                // Build and compile model
                IModel model;
                if (Directory.Exists(modelPath) && Directory.GetDirectories(modelPath).Length > 0)
                {
                    model = keras.models.load_model(AIConfig.TrainingModelFullURL);
                }
                else
                {
                    var input = keras.Input(shape: AIConfig.ImageShape);

                    var x = keras.layers.Conv2D(32, 3, activation: keras.activations.Relu).Apply(input);
                    x = keras.layers.MaxPooling2D().Apply(x);
                    x = keras.layers.Conv2D(64, 3, activation: keras.activations.Relu).Apply(x);
                    x = keras.layers.MaxPooling2D().Apply(x);
                    x = keras.layers.Flatten().Apply(x);
                    x = keras.layers.Dense(128, activation: keras.activations.Relu).Apply(x);
                    var output = keras.layers.Dense(4, activation: keras.activations.Sigmoid).Apply(x); // xmin, ymin, xmax, ymax

                    model = keras.Model(input, output, name: AIConfig.TrainingModelName);
                }

                model.summary();
                model.compile(
                    optimizer: keras.optimizers.Adam(learning_rate: 0.001f),
                    loss: keras.losses.MeanSquaredError(),
                    metrics: new[] { "mean_absolute_error" });


                var trainData = LoadTFRecord(trainTfPath);

                NDArray xTrain, yTrain;
                Create_ND_array(trainData, out xTrain, out yTrain);

                if (xTrain == null || yTrain == null)
                    return $"Training failed:\r\nxTrain or yTrain is null";


                // Train and Validate         
                ICallback history = model.fit(xTrain, yTrain,
                                    epochs: epochs,
                                    use_multiprocessing: true,
                                    workers: Environment.ProcessorCount,  // Use all available threads
                                    max_queue_size: 32,                   // Increase input queue size
                                    shuffle: true,                       // Ensures better generalizatio
                                    validation_split: AIConfig.ValidationDataSplit);

                // Save   
                AIConfig.TrainingModelFullURL = AIConfig.TrainingModelFullURL;
                model.save(modelPath);                      
            

                // Report
                float loss = history.history["loss"].Last();
                float mae = history.history["mean_absolute_error"].Last();
                float val_loss = history.history["val_loss"].Last();
                float val_mae = history.history["val_mean_absolute_error"].Last();

                return $"Training completed:\r\nTrain Loss: {loss:F4}\r\nTrain MAE: {mae:F4}\r\nValidation Loss: {val_loss:F4}\r\nValidation MAE: {val_mae:F4}";
            }
            catch (Exception ex)
            {
                return $"Training failed:\r\n{ex}";
            }
        }

        


        private static void Create_ND_array(List<Example> trainData, out NDArray xTrain, out NDArray yTrain)
        {
            xTrain = null;
            yTrain = null;

            try
            {
                int imageChannels = 1;
                int imageSize = AIConfig.ImageShape * AIConfig.ImageShape;

                var images = new List<float[]>();
                var bboxes = new List<float[]>(); // store bbox as float[4] per example: [xmin, ymin, xmax, ymax]

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

                    float xmin = xmins[0];
                    float ymin = ymins[0];
                    float xmax = xmaxs[0];
                    float ymax = ymaxs[0];

                    using var ms = new MemoryStream(imageBytes);
                    using var bmp = new Bitmap(ms);
                    using var resized = new Bitmap(bmp, new Size(AIConfig.ImageShape, AIConfig.ImageShape));

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
                    bboxes.Add(new float[] { xmin, ymin, xmax, ymax });
                }

                if (images.Count == 0)
                    return;

                int sampleCount = images.Count;
                var reshapedImages = new float[sampleCount, AIConfig.ImageShape, AIConfig.ImageShape, imageChannels];
                var bboxArray = new float[sampleCount, 4];

                for (int i = 0; i < sampleCount; i++)
                {
                    for (int row = 0; row < AIConfig.ImageShape; row++)
                    {
                        for (int col = 0; col < AIConfig.ImageShape; col++)
                        {
                            reshapedImages[i, row, col, 0] = images[i][row * AIConfig.ImageShape + col];
                        }
                    }

                    for (int k = 0; k < 4; k++)
                    {
                        bboxArray[i, k] = bboxes[i][k];
                    }
                }

                xTrain = np.array(reshapedImages);
                yTrain = np.array(bboxArray);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Create_ND_array] Error: {ex.Message}");
                xTrain = null;
                yTrain = null;
            }
        }



        private static List<Example> LoadTFRecord(string tfRecordPath)
        {
            List<Example> examples = new List<Example>();

            using (var file = File.OpenRead(tfRecordPath))
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

                    using (var ms = new MemoryStream(data))
                    {
                        var example = Serializer.Deserialize<Example>(ms);
                        examples.Add(example);
                    }
                }
            }

            return examples;
        }

    }
}
