using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace AICircleDetector.AI
{
    public static class Trainer
    {
        public static void Train(string folderGUIDPath)
        {
            // Resolve dataset paths
            string imageDir = Path.Combine(folderGUIDPath, AIConfig.ImageFolderName);
            string annotationDir = Path.Combine(folderGUIDPath, AIConfig.AnnotationsFolderName);

            if (!Directory.Exists(imageDir) || !Directory.Exists(annotationDir))
            {
                Console.WriteLine($"❌ Invalid dataset path: {folderGUIDPath}");
                return;
            }

            Console.WriteLine($"🧠 Loading training data from: {folderGUIDPath}");

            // --- Step 1: Load images + annotations ---
            var imageFiles = Directory.GetFiles(imageDir, "*.png");
            var xData = new List<float[]>();
            var yData = new List<float[]>();

            foreach (var imgPath in imageFiles)
            {
                var imageName = Path.GetFileName(imgPath);
                var annotationPath = Path.Combine(annotationDir, Path.GetFileNameWithoutExtension(imageName) + ".json");

                if (!File.Exists(annotationPath))
                    continue;

                // Load image
                using var bmp = new Bitmap(imgPath);
                var resized = new Bitmap(bmp, AIConfig.ImageSize);

                var imgArray = new float[AIConfig.ImageSize.Width * AIConfig.ImageSize.Height * 3];
                int index = 0;
                for (int row = 0; row < resized.Height; row++)
                {
                    for (int col = 0; col < resized.Width; col++)
                    {
                        var c = resized.GetPixel(col, row);
                        imgArray[index++] = c.R / 255f;
                        imgArray[index++] = c.G / 255f;
                        imgArray[index++] = c.B / 255f;
                    }
                }

                xData.Add(imgArray);

                // Load annotation and build label vector
                var json = File.ReadAllText(annotationPath);
                if (string.IsNullOrEmpty(json))
                    return;

                AnnotationFile? annotation = JsonConvert.DeserializeObject<AnnotationFile>(json);
                if (annotation == null)
                    return;

                float[] labelVector = new float[1 + (AIConfig.MaxCircles * 4)];
                int circleCount = annotation.Objects.Count;
                labelVector[0] = circleCount;

                for (int i = 0; i < Math.Min(circleCount, AIConfig.MaxCircles); i++)
                {
                    var c = annotation.Objects[i];
                    int baseIndex = 1 + (i * 4);
                    labelVector[baseIndex] = c.XMin / AIConfig.ImageSize.Width;
                    labelVector[baseIndex + 1] = c.YMin / AIConfig.ImageSize.Height;
                    labelVector[baseIndex + 2] = c.XMax / AIConfig.ImageSize.Width;
                    labelVector[baseIndex + 3] = c.YMax / AIConfig.ImageSize.Height;
                }

                yData.Add(labelVector);
            }

            var x = np.array(xData.ToArray());
            var y = np.array(yData.ToArray());

            Console.WriteLine($"📊 Loaded {x.shape[0]} samples.");

            // --- Step 2: Build CNN Model ---
            var g = tf.Graph().as_default();
            var session = tf.Session(g);

            var x_input = tf.placeholder(tf.float32, shape: new Shape(-1, AIConfig.ImageSize.Width, AIConfig.ImageSize.Height, 3));
            var y_true = tf.placeholder(tf.float32, shape: new Shape(-1, 1 + (AIConfig.MaxCircles * 4)));


            var model = new Sequential(new Tensorflow.Keras.ArgsDefinition.SequentialArgs());
            // Conv2D Layer mit Args
            var conv1Args = new Tensorflow.Keras.ArgsDefinition.Conv2DArgs
            {
                Filters = 32,
                KernelSize = (3, 3),
                Activation = new relu(),
                InputShape = new Shape(AIConfig.ImageSize.Width, AIConfig.ImageSize.Height, 3)
            };

            model.Add(new Conv2D(conv1Args)); model.Add(new MaxPooling2D(pool_size: 2, strides: 2));
            model.Add(new Conv2D(64, kernel_size: 3, activation: tf.nn.relu));
            model.Add(new MaxPooling2D(pool_size: 2, strides: 2));
            model.Add(new Flatten());
            model.Add(new Dense(128, activation: tf.nn.relu));
            model.Add(new Dense(1 + (AIConfig.MaxCircles * 4)));



            // Loss: mean squared error for regression targets
            var loss = tf.reduce_mean(tf.square(output - y_true));
            var optimizer = tf.train.AdamOptimizer(0.001f).minimize(loss);

            // --- Step 3: Train ---
            session.run(tf.global_variables_initializer());

            int epochs = 10;
            int batchSize = 16;

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                float epochLoss = 0;
                for (int i = 0; i < x.shape[0]; i += batchSize)
                {
                    var end = Math.Min(i + batchSize, (int)x.shape[0]);
                    var batchX = x[new Slice(i, end)];
                    var batchY = y[new Slice(i, end)];

                    var (_, lossVal) = session.run((optimizer, loss),
                        (x_input, batchX),
                        (y_true, batchY));

                    epochLoss += lossVal;
                }

                Console.WriteLine($"Epoch {epoch}/{epochs} — Loss: {epochLoss / (x.shape[0] / batchSize):F4}");
            }

            // --- Step 4: Save Model ---
            string savePath = Path.Combine(AIConfig.TrainingModelFullURL, AIConfig.TrainingModelName);
            Directory.CreateDirectory(AIConfig.TrainingModelFullURL);
            var saver = tf.train.Saver();
            saver.save(session, savePath);

            Console.WriteLine($"💾 Model saved at: {savePath}");
        }
       
    }
}
