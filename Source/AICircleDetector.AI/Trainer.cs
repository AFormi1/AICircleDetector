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


            string modelPath = AIConfig.TrainingModelFullURL;
            IModel model;

            if (Directory.Exists(modelPath) && Directory.GetDirectories(modelPath).Length > 0)
            {
                // Modell laden, falls vorhanden
                model = tf.keras.models.load_model(modelPath);
            }
            else
            {
                // Input Tensor
                var input = tf.keras.Input(shape: (AIConfig.ImageSize.Width, AIConfig.ImageSize.Height, 3));

                // CNN
                var cnn = tf.keras.layers.Conv2D(32, 3, activation: tf.keras.activations.Relu).Apply(input);
                cnn = tf.keras.layers.MaxPooling2D().Apply(x);
                cnn = tf.keras.layers.Conv2D(64, 3, activation: tf.keras.activations.Relu).Apply(x);
                cnn = tf.keras.layers.MaxPooling2D().Apply(x);
                cnn = tf.keras.layers.Flatten().Apply(x);
                cnn = tf.keras.layers.Dense(128, activation: tf.keras.activations.Relu).Apply(x);

                // Output für Bounding Boxes
                var bbox_output = tf.keras.layers.Dense(AIConfig.MaxCircles * 4, activation: tf.keras.activations.Sigmoid).Apply(x);
                bbox_output = tf.keras.layers.Reshape((AIConfig.MaxCircles, 4)).Apply(bbox_output);

                // Modell erstellen
                model = tf.keras.Model(input, bbox_output);
            }


            // Modell kompilieren
            model.compile(
                    optimizer: tf.keras.optimizers.Adam(),
                    loss: tf.keras.losses.MeanSquaredError(),
                    metrics: new[] { "mean_absolute_error" }
            );

            // Trainieren
            model.fit(x, y, batch_size: 16, epochs: 10);


            // --- Step 4: Save Model ---
            string savePath = Path.Combine(AIConfig.TrainingModelFullURL, AIConfig.TrainingModelName);
            Directory.CreateDirectory(AIConfig.TrainingModelFullURL);
            var saver = tf.train.Saver();
            saver.save(session, savePath);

            Console.WriteLine($"💾 Model saved at: {savePath}");
        }

    }
}
