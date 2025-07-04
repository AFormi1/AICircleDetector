using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;
using ProtoBuf;
using System.Drawing;
using System.Reflection.Metadata.Ecma335;
using OneOf.Types;

namespace AICircleDetector.AI
{
    public static class TrainerAndValidator
    {
        public static AIResult Train(CancellationToken cancellationToken, string basepath)
        {
            AIResult result = new();

            try
            {
                string trainTFRecordPath = Path.Combine(basepath, AIConfig.TrainDataName);

                // Step 1: Load the TFRecord files
                var trainData = LoadTFRecord(trainTFRecordPath);

                // Step 2: Train the model with the trainData
                result = TrainModel(cancellationToken, trainData, basepath);

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

        public static AIResult Validate(CancellationToken cancellationToken, string basepath)
        {
            AIResult result = new();

            try
            {
                //Step 1: load the model if exists
                if (!Path.Exists(AIConfig.TrainingModelFullURL))
                {
                    return new AIResult
                    {
                        Success = false,
                        Message = "Validation canceld, no pre-trained model found!"
                    };
                }

                var modelPath = AIConfig.TrainingModelFullURL;
                var model = keras.models.load_model(modelPath);

                model.summary();

                //model.compile(optimizer: keras.optimizers.Adam(),
                //              loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                //              metrics: new[] { "accuracy" });

                model.compile(
                    optimizer: keras.optimizers.Adam(),
                    loss: keras.losses.MeanSquaredError(), // or MeanAbsoluteError
                    metrics: new[] { "mean_absolute_error" }); // or "mse"

                //Step 1: Load the TFRecord files
                string valTFRecordPath = Path.Combine(basepath, AIConfig.ValDataName);

                if (cancellationToken.IsCancellationRequested)
                    return result;

                var valData = LoadTFRecord(valTFRecordPath);

                //Step 2: Create the ND array 
                if (cancellationToken.IsCancellationRequested)
                    return result;

                var validationResult = Create_ND_array(valData, out NDArray xVal, out NDArray yVal, cancellationToken);

                if (!validationResult.Success)
                    return validationResult;

                // Step 3: validate the model with the validationData
                if (cancellationToken.IsCancellationRequested)
                    return result;

                yVal = yVal.reshape(new Shape(yVal.shape[0])); // ensure 1D labels

                var evalResult = model.evaluate(xVal, yVal, verbose: 1);

                float valLoss = evalResult.ContainsKey("loss") ? evalResult["loss"] : float.NaN;
                float valMAE = evalResult.ContainsKey("mean_absolute_error") ? evalResult["mean_absolute_error"] : float.NaN;

                return new AIResult
                {
                    Success = true,
                    Message = "Validation completed successfully.",
                    Loss = valLoss,
                    MAE = valMAE
                };

            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                result.Success = false;
                result.Message = ex.Message;
            }

            return result;
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




        public static AIResult TrainModel(CancellationToken cancellationToken, List<Example> trainData, string basePath)
        {
            NDArray xTrain, yTrain;

            var result = Create_ND_array(trainData, out xTrain, out yTrain, cancellationToken);

            if (!result.Success)
                return result;

            AIResult trainingResult = SetupAndTrainModel(xTrain, yTrain, basePath);

            return trainingResult;
        }

        private static AIResult Create_ND_array(List<Example> trainData, out NDArray xTrain, out NDArray yTrain, CancellationToken cancellationToken)
        {
            int imageChannels = 1;
            int imageSize = AIConfig.TrainerShapeSize * AIConfig.TrainerShapeSize;

            var images = new List<float[]>();
            var labels = new List<int>();

            foreach (var example in trainData)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    xTrain = null;
                    yTrain = null;
                    return new AIResult { Success = false, Message = "Training cancelled." };
                }

                var featureMap = example.Features?.feature;
                if (featureMap == null ||
                    !featureMap.TryGetValue("image", out var imageFeature) ||
                    !featureMap.TryGetValue("label", out var labelFeature))
                    continue;

                var imageBytes = imageFeature.BytesList?.Values?[0];
                var label = labelFeature.Int64List?.Values?[0] ?? 0;

                if (imageBytes == null || imageBytes.Length == 0)
                    continue;

                using var ms = new MemoryStream(imageBytes);
                using var bmp = new Bitmap(ms);
                using var resized = new Bitmap(bmp, new Size(AIConfig.TrainerShapeSize, AIConfig.TrainerShapeSize));

                float[] imageFloats = new float[imageSize];
                for (int y = 0; y < AIConfig.TrainerShapeSize; y++)
                {
                    for (int x = 0; x < AIConfig.TrainerShapeSize; x++)
                    {
                        Color pixel = resized.GetPixel(x, y);
                        float gray = (pixel.R + pixel.G + pixel.B) / 3f / 255f;
                        imageFloats[y * AIConfig.TrainerShapeSize + x] = gray;
                    }
                }

                images.Add(imageFloats);
                labels.Add((int)label);
            }

            if (images.Count == 0)
            {
                xTrain = null;
                yTrain = null;
                return new AIResult { Success = false, Message = "No valid training data found." };
            }

            int sampleCount = images.Count;
            var reshapedImages = new float[sampleCount, AIConfig.TrainerShapeSize, AIConfig.TrainerShapeSize, imageChannels];

            for (int i = 0; i < sampleCount; i++)
            {
                for (int row = 0; row < AIConfig.TrainerShapeSize; row++)
                {
                    for (int col = 0; col < AIConfig.TrainerShapeSize; col++)
                    {
                        reshapedImages[i, row, col, 0] = images[i][row * AIConfig.TrainerShapeSize + col];
                    }
                }
            }

            xTrain = np.array(reshapedImages);
            yTrain = np.array(labels.ToArray());

            return new AIResult { Success = true };
        }


        public static AIResult SetupAndTrainModel(NDArray xTrain, NDArray yTrain, string basepath)
        {
            // Reshape labels to 1D
            yTrain = yTrain.reshape(new Shape(yTrain.shape[0]));

            var layers = keras.layers;

            var inputs = keras.Input(shape: (AIConfig.TrainerShapeSize, AIConfig.TrainerShapeSize, 1), name: "input_layer");

            // Build the model
            var x = layers.Conv2D(32, kernel_size: 3, activation: "relu", padding: "same").Apply(inputs);
            x = layers.MaxPooling2D(pool_size: 2).Apply(x);

            x = layers.Conv2D(64, kernel_size: 3, activation: "relu", padding: "same").Apply(x);
            x = layers.MaxPooling2D(pool_size: 2).Apply(x);

            x = layers.Flatten().Apply(x);
            x = layers.Dense(128, activation: "relu").Apply(x);
            x = layers.Dropout(0.5f).Apply(x);

            // Output layer: 10 classes as you had before
            var outputs = layers.Dense(1).Apply(x);

            var model = keras.Model(inputs, outputs, name: "CircleDetection");

            model.summary();

            //model.compile(optimizer: keras.optimizers.Adam(),
            //              loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
            //              metrics: new[] { "accuracy" });

            model.compile(
                    optimizer: keras.optimizers.Adam(),
                    loss: keras.losses.MeanSquaredError(), // or MeanAbsoluteError
                    metrics: new[] { "mean_absolute_error" } // or "mse"
                );

            Console.WriteLine("Starting training...");

            // Train the model using your data (make sure xTrain is normalized [0,1])
            model.fit(xTrain, yTrain,
                      batch_size: 64,
                      epochs: 100);

            Console.WriteLine("Training complete.");

            // Save the model
            AIConfig.TrainingModelFullURL = AIConfig.TrainingModelFullURL;

            model.save(AIConfig.TrainingModelFullURL);

            var evalResult = model.evaluate(xTrain, yTrain, verbose: 0);

            // Safely retrieve loss and accuracy by key
            float valLoss = evalResult.ContainsKey("loss") ? evalResult["loss"] : float.NaN;
            float valMAE = evalResult.ContainsKey("mean_absolute_error") ? evalResult["mean_absolute_error"] : float.NaN;

            var result = new AIResult
            {
                Success = true,
                Message = "Training completed successfully.",
                Loss = valLoss,
                MAE = valMAE
            };

            return result;
        }


    }

}
