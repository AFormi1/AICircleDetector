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
using Tensorflow.Data;

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
                    model = keras.models.load_model(modelPath);
                }
                else
                {
                    Tensor input = keras.Input(shape: (AIConfig.ImageShape, AIConfig.ImageShape, 1));

                    // Feature extraction
                    var x = keras.layers.Conv2D(32, 3, activation: keras.activations.Relu).Apply(input);
                    x = keras.layers.MaxPooling2D().Apply(x);
                    x = keras.layers.Conv2D(64, 3, activation: keras.activations.Relu).Apply(x);
                    x = keras.layers.MaxPooling2D().Apply(x);
                    x = keras.layers.Flatten().Apply(x);
                    x = keras.layers.Dense(128, activation: keras.activations.Relu).Apply(x);

                    // Outputs
                    var bbox_output = keras.layers.Dense(AIConfig.MaxCircles * 4, activation: keras.activations.Sigmoid).Apply(x);
                    bbox_output = keras.layers.Reshape((AIConfig.MaxCircles, 4)).Apply(bbox_output);

                    var count_output = keras.layers.Dense(1, activation: keras.activations.Sigmoid).Apply(x);

                    var outputs = new Tensors(bbox_output, count_output);

                    model = keras.Model(input, outputs);
                }

                model.summary();

                // Use your combined loss class here
                var combinedLoss = new CombinedLoss();

                model.compile(
                    optimizer: keras.optimizers.Adam(),
                    loss: combinedLoss,
                    metrics: new[] { "mean_absolute_error" }
                );

                // Load data
                List<Example> trainData = LoadTFRecord(trainTfPath);
                List<Example> validationData = LoadTFRecord(validationTfPath);

                NDArray xTrain, bboxTrain, countTrain;
                NDArray xValidate, bboxValidate, countValidate;

                Create_ND_array(trainData, out xTrain, out bboxTrain, out countTrain);
                Create_ND_array(validationData, out xValidate, out bboxValidate, out countValidate);

                if (
                    xTrain == null || bboxTrain == null || countTrain == null ||
                    xValidate == null || bboxValidate == null || countValidate == null
                )
                    return $"Training failed:\r\nOne of the required NDArrays is null";

                // 1. Flatten bbox to match model output
                var flatBBoxTrain = tf.constant(bboxTrain.reshape(new Shape(bboxTrain.shape[0], -1))); // [batch, maxCircles*4]
                var countTrainT = tf.constant(countTrain); // [batch, 1] or [batch]

                // 2. Create individual datasets
                var dsX = new TensorSliceDataset(tf.constant(xTrain));           // input
                var dsBBox = new TensorSliceDataset(flatBBoxTrain);              // output 1
                var dsCount = new TensorSliceDataset(countTrainT);               // output 2

                // 3. Combine (bbox, count) -> label tuple
                var dsLabel = tf.data.Dataset.zip(dsBBox, dsCount);              // (bbox, count)

                // 4. Combine (input, label)
                var trainDataset = tf.data.Dataset.zip(dsX, dsLabel)             // ((x), (bbox, count))
                    .shuffle(1000)
                    .batch(batchSize);


                var flatBBoxVal = tf.constant(bboxValidate.reshape(new Shape(bboxValidate.shape[0], -1)));
                var countValT = tf.constant(countValidate);

                var dsXVal = new TensorSliceDataset(tf.constant(xValidate));
                var dsBBoxVal = new TensorSliceDataset(flatBBoxVal);
                var dsCountVal = new TensorSliceDataset(countValT);

                var dsLabelVal = tf.data.Dataset.zip(dsBBoxVal, dsCountVal);
                var validationDataset = tf.data.Dataset.zip(dsXVal, dsLabelVal)
                    .batch(batchSize);


                // Train the model
                ICallback history = model.fit(
                    trainDataset,
                    epochs: epochs,
                    validation_data: validationDataset,
                    shuffle: true,
                    verbose: 1
                );

                // Save model
                model.save(modelPath);

                // Extract metrics from history (adjust keys if needed)
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










        private static void Create_ND_array(List<Example> trainData, out NDArray xTrain, out NDArray bboxTrain, out NDArray countTrain)
        {
            xTrain = null;
            bboxTrain = null;
            countTrain = null;

            try
            {
                int imageChannels = 1; // Grayscale
                int imageSize = AIConfig.ImageShape * AIConfig.ImageShape;
                int maxBoxes = AIConfig.MaxCircles; // max boxes per image

                var images = new List<float[]>();
                var allBBoxes = new List<float[,]>();
                var counts = new List<float>(); // for circle counts

                foreach (var example in trainData)
                {
                    var featureMap = example.Features?.feature;
                    if (featureMap == null ||
                        !featureMap.TryGetValue("image/encoded", out var imageFeature) ||
                        !featureMap.TryGetValue("image/circle_count", out var countFeature) ||
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

                    // Extract circle count value as float
                    var countValue = countFeature.FloatList?.Values?.FirstOrDefault() ?? 0f;
                    counts.Add(countValue);

                    // Load and resize image to AIConfig.ImageShape x AIConfig.ImageShape
                    using var ms = new MemoryStream(imageBytes);
                    using var bmp = new Bitmap(ms);
                    using var resized = new Bitmap(bmp, new Size(AIConfig.ImageShape, AIConfig.ImageShape));

                    // Convert to grayscale float array [0..1]
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

                    // Pack bounding boxes into padded array maxBoxes x 4
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
                var countArray = new float[sampleCount];

                // Convert 1D images to 4D array & copy bboxes and counts
                for (int i = 0; i < sampleCount; i++)
                {
                    for (int row = 0; row < AIConfig.ImageShape; row++)
                    {
                        for (int col = 0; col < AIConfig.ImageShape; col++)
                        {
                            reshapedImages[i, row, col, 0] = images[i][row * AIConfig.ImageShape + col];
                        }
                    }

                    for (int b = 0; b < maxBoxes; b++)
                    {
                        for (int k = 0; k < 4; k++)
                        {
                            bboxArray[i, b, k] = allBBoxes[i][b, k];
                        }
                    }

                    countArray[i] = counts[i];
                }

                xTrain = np.array(reshapedImages);
                bboxTrain = np.array(bboxArray);
                countTrain = np.array(countArray);    
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Create_ND_array] Error: {ex.Message}");
                xTrain = null;
                bboxTrain = null;
                countTrain = null;
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
