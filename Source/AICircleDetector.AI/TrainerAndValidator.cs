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
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;

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

                var validationResult = Create_ND_array(valData, out NDArray xVal, out NDArray yValCount, out NDArray yValDiameters, cancellationToken);

                if (!validationResult.Success)
                    return validationResult;

                // Step 3: validate the model with the validationData
                if (cancellationToken.IsCancellationRequested)
                    return result;

                yValCount = yValCount.reshape(new Shape(yValCount.shape[0])); // ensure 1D labels

                var evalResult = model.evaluate(xVal, yValCount, verbose: 1);

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


        private static (byte[] imageBytes, int count, int[] diameters)? ParseExample(Example example, int maxCircles)
        {
            var featureMap = example.Features?.feature;
            if (featureMap == null)
                return null;

            if (!featureMap.TryGetValue("image/encoded", out var imageFeature) ||
                !featureMap.TryGetValue("image/object/bbox/xmin", out var xminFeature) ||
                !featureMap.TryGetValue("image/object/bbox/ymin", out var yminFeature) ||
                !featureMap.TryGetValue("image/object/bbox/xmax", out var xmaxFeature) ||
                !featureMap.TryGetValue("image/object/bbox/ymax", out var ymaxFeature))
                return null;

            var imageBytes = imageFeature.BytesList?.Values?[0];
            if (imageBytes == null || imageBytes.Length == 0)
                return null;

            var xmin = xminFeature?.FloatList?.Values?.ToList() ?? new List<float>();
            var ymin = yminFeature?.FloatList?.Values?.ToList() ?? new List<float>();
            var xmax = xmaxFeature?.FloatList?.Values?.ToList() ?? new List<float>();
            var ymax = ymaxFeature?.FloatList?.Values?.ToList() ?? new List<float>();

            int boxCount = Math.Min(Math.Min(xmin.Count, ymin.Count), Math.Min(xmax.Count, ymax.Count));
            boxCount = Math.Min(boxCount, maxCircles);

            // Calculate diameters (widths or heights of boxes)
            int[] diameters = new int[maxCircles];
            for (int i = 0; i < boxCount; i++)
            {
                var x = xmax[i] - xmin[i];  // normalized width
                var y = ymax[i] - ymin[i];  // normalized height

                var avg = (x + y) / 2.0f;

                diameters[i] = (int)Math.Round(avg * AIConfig.TrainerShapeSize, 0);
            }
            // Remaining diameters default to zero

            return (imageBytes, boxCount, diameters);
        }



        public static AIResult TrainModel(CancellationToken cancellationToken, List<Example> trainData, string basePath)
        {
            NDArray xTrain, yTrainCount, yTrainDiameters;

            var result = Create_ND_array(trainData, out xTrain, out yTrainCount, out yTrainDiameters, cancellationToken);

            if (!result.Success)
                return result;

            AIResult trainingResult = SetupAndTrainModel(xTrain, yTrainCount, yTrainDiameters, basePath);

            return trainingResult;
        }


        private static AIResult Create_ND_array(List<Example> trainData, out NDArray xTrain, out NDArray yTrainCounts, out NDArray yTrainDiameters, CancellationToken cancellationToken)
        {
            int imageChannels = 1;
            int imageSize = AIConfig.TrainerShapeSize * AIConfig.TrainerShapeSize;
         
            var images = new List<int[]>();
            var counts = new List<int>();
            var diametersList = new List<int[]>();  // List of diameter arrays (padded)

            foreach (var example in trainData)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    xTrain = null;
                    yTrainCounts = null;
                    yTrainDiameters = null;
                    return new AIResult { Success = false, Message = "Training cancelled." };
                }

                //will return: {byte[] image, CircleCount, int[] diameters }
                var parsed = ParseExample(example, AIConfig.MaxCircles);
                if (parsed == null)
                    continue;

                var (imageBytes, count, paddedDiameters) = parsed.Value;

                using var ms = new MemoryStream(imageBytes);
                using var bmp = new Bitmap(ms);
                using var resized = new Bitmap(bmp, new Size(AIConfig.TrainerShapeSize, AIConfig.TrainerShapeSize));

                int[] imageFloats = new int[imageSize];
                for (int y = 0; y < AIConfig.TrainerShapeSize; y++)
                {
                    for (int x = 0; x < AIConfig.TrainerShapeSize; x++)
                    {
                        Color pixel = resized.GetPixel(x, y);
                        int gray = (int)Math.Round((pixel.R + pixel.G + pixel.B) / 3f / 255f, 0);
                        imageFloats[y * AIConfig.TrainerShapeSize + x] = gray;
                    }
                }

                images.Add(imageFloats);
                counts.Add(count);
                diametersList.Add(paddedDiameters);
            }

            if (images.Count == 0)
            {
                xTrain = null;
                yTrainCounts = null;
                yTrainDiameters = null;
                return new AIResult { Success = false, Message = "No valid training data found." };
            }

            int sampleCount = images.Count;
            var reshapedImages = new int[sampleCount, AIConfig.TrainerShapeSize, AIConfig.TrainerShapeSize, imageChannels];
            var labelsCounts = new int[sampleCount];
            var labelsDiameters = new int[sampleCount, AIConfig.MaxCircles];

            for (int i = 0; i < sampleCount; i++)
            {
                for (int row = 0; row < AIConfig.TrainerShapeSize; row++)
                {
                    for (int col = 0; col < AIConfig.TrainerShapeSize; col++)
                    {
                        reshapedImages[i, row, col, 0] = images[i][row * AIConfig.TrainerShapeSize + col];
                    }
                }
                labelsCounts[i] = counts[i];

                for (int d = 0; d < AIConfig.MaxCircles; d++)
                    labelsDiameters[i, d] = diametersList[i][d];
            }

            xTrain = np.array(reshapedImages);
            yTrainCounts = np.array(labelsCounts);
            yTrainDiameters = np.array(labelsDiameters);

            return new AIResult { Success = true };
        }


        private static AIResult SetupAndTrainModel(NDArray xTrain, NDArray yTrainCount, NDArray yTrainDiameters, string basepath)
        {
            // Ensure proper label shape
            yTrainCount = yTrainCount.reshape(new Shape(yTrainCount.shape[0], 1));
            yTrainDiameters = yTrainDiameters.reshape(new Shape(yTrainDiameters.shape[0], AIConfig.MaxCircles));

            var layers = keras.layers;
            var shape = new Shape(AIConfig.TrainerShapeSize, AIConfig.TrainerShapeSize, 1);

            var inputs = keras.Input(shape, name: "input_layer");

            // CNN Backbone 
            var input = keras.Input(shape: shape);

            var x = layers.Conv2D(32, new Shape(3, 3), null, padding: "same").Apply(input);
            x = layers.MaxPooling2D(pool_size: new Shape(2, 2)).Apply(x);

            x = layers.BatchNormalization().Apply(x);
            x = layers.LeakyReLU().Apply(x);
            x = layers.MaxPooling2D(2).Apply(x);

            x = layers.Conv2D(64, new Shape(3, 3), null, padding: "same").Apply(input);
            x = layers.BatchNormalization().Apply(x);
            x = layers.LeakyReLU().Apply(x);
            x = layers.MaxPooling2D(2).Apply(x);

            x = layers.Flatten().Apply(x);
            x = layers.Dense(128).Apply(x);
            x = layers.LeakyReLU().Apply(x);
            x = layers.Dropout(0.5f).Apply(x);

            // Output branches
            x = keras.layers.Dense(128, activation: keras.activations.Relu).Apply(x);

            var countOutput = layers.Dense(1).Apply(x);
            var diamOutput = layers.Dense(AIConfig.MaxCircles).Apply(x);

            // Model
            IModel model;
            if (Path.Exists(AIConfig.TrainingModelFullURL))            
                model = keras.models.load_model(AIConfig.TrainingModelFullURL);    
            else            
                model = keras.Model(inputs, new Tensors(countOutput, diamOutput), name: "CircleRegressionModel");


            model.summary();

            model.compile(
                optimizer: keras.optimizers.Adam(),
                loss: keras.losses.MeanSquaredError(),
                metrics: new[] { "mean_absolute_error" }
            );

            //todo - continue here with current exception, maybe problem with int[] vs float[]

            //var yTrainCountFloat = yTrainCount.astype(np.float32);
            //var yTrainDiametersFloat = yTrainDiameters.astype(np.float32);

            //var combinedLabels = np.concatenate(new NDArray[] { yTrainCountFloat.reshape(new Shape(-1, 1)), yTrainDiametersFloat }, axis: 1);


            var combinedLabels = np.concatenate(new NDArray[] { yTrainCount.reshape(new Shape(-1, 1)), yTrainDiameters }, axis: 1);

            // Train
            model.fit(
                xTrain,
                combinedLabels,
                batch_size: 128,
                epochs: 200,
                shuffle: true,
                use_multiprocessing: true,
                workers: Environment.ProcessorCount,
                max_queue_size: 32
            );

            Console.WriteLine("Training complete.");

            // Save the model
            model.save(AIConfig.TrainingModelFullURL);           

            //float lossCount = eval.ContainsKey("count_output_loss") ? eval["count_output_loss"] : float.NaN;
            //float lossDiam = eval.ContainsKey("diameter_output_loss") ? eval["diameter_output_loss"] : float.NaN;
            //float maeCount = eval.ContainsKey("count_output_mean_absolute_error") ? eval["count_output_mean_absolute_error"] : float.NaN;
            //float maeDiam = eval.ContainsKey("diameter_output_mean_absolute_error") ? eval["diameter_output_mean_absolute_error"] : float.NaN;

            return new AIResult
            {
                Success = true,
                Message = "Training completed successfully."//,
                //Loss = lossCount + lossDiam,
                //MAE = (maeCount + maeDiam) / 2.0f
            };
        }



        //private static AIResult SetupAndTrainModel(NDArray xTrain, NDArray yTrainCount, NDArray yTrainDiameters, string basepath)
        //{
        //    // Reshape labels to 1D
        //    yTrainCount = yTrainCount.reshape(new Shape(yTrainCount.shape[0]));

        //    var layers = keras.layers;

        //    var inputs = keras.Input(shape: (AIConfig.TrainerShapeSize, AIConfig.TrainerShapeSize, 1), name: "input_layer");

        //    // Build the model

        //    var x = layers.Conv2D(32, kernel_size: 3, activation: keras.activations.Relu, padding: "same").Apply(inputs);
        //    x = layers.MaxPooling2D(pool_size: 2).Apply(x);

        //    x = layers.Conv2D(64, kernel_size: 3, activation: keras.activations.Relu, padding: "same").Apply(x);
        //    x = layers.MaxPooling2D(pool_size: 2).Apply(x);

        //    x = layers.Flatten().Apply(x);
        //    x = keras.layers.Dense(128, activation: keras.activations.Relu).Apply(x);
        //    x = keras.layers.Dropout(0.5f).Apply(x);

        //    // Output layer: 10 classes as you had before
        //    var outputs = layers.Dense(1).Apply(x);


        //    // 1. Build and compile your model
        //    IModel model;
        //    if (Path.Exists(AIConfig.TrainingModelFullURL))
        //    {
        //        model = keras.models.load_model(AIConfig.TrainingModelFullURL);
        //    }
        //    else
        //    {
        //        model = keras.Model(inputs, outputs, name: "CircleDetection");
        //    }
        //    model.summary();

        //    model.compile(
        //        optimizer: keras.optimizers.Adam(),
        //        loss: keras.losses.MeanSquaredError(),
        //        metrics: new[] { "mean_absolute_error" }
        //    );

        //    int batch = 128;
        //    int epochs = 200;

        //    // 4. Train
        //    model.fit(
        //        xTrain, yTrainCount,
        //        batch_size: batch,
        //        epochs: epochs,
        //        use_multiprocessing: true,
        //        workers: Environment.ProcessorCount,  // Use all available threads
        //        max_queue_size: 32,                   // Increase input queue size
        //        shuffle: true                         // Ensures better generalization
        //    );

        //    Console.WriteLine("Training complete.");

        //    // Save the model
        //    AIConfig.TrainingModelFullURL = AIConfig.TrainingModelFullURL;

        //    model.save(AIConfig.TrainingModelFullURL);

        //    var evalResult = model.evaluate(xTrain, yTrainCount, verbose: 0, use_multiprocessing: true,
        //        workers: Environment.ProcessorCount,  // Use all available threads
        //        max_queue_size: 32); //Increase input queue size

        //    // Safely retrieve loss and accuracy by key
        //    float valLoss = evalResult.ContainsKey("loss") ? evalResult["loss"] : float.NaN;
        //    float valMAE = evalResult.ContainsKey("mean_absolute_error") ? evalResult["mean_absolute_error"] : float.NaN;

        //    var result = new AIResult
        //    {
        //        Success = true,
        //        Message = "Training completed successfully.",
        //        Loss = valLoss,
        //        MAE = valMAE
        //    };

        //    return result;
        //}


    }

}
