using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Text;
using Newtonsoft.Json;

namespace AICircleDetector.AI
{
    public static class TrainingDataBuilder
    {
        private static readonly Random rnd = new Random();

        public static bool GenerateTrainingData(int numSamples = 500)
        {
            // Create unique batch folder
            string batchId = Guid.NewGuid().ToString();
            string baseDir = Path.Combine(AIConfig.TrainingFolderName, batchId);

            string imageDir = Path.Combine(baseDir, AIConfig.ImageFolderName);
            string annotationDir = Path.Combine(baseDir, AIConfig.AnnotationsFolderName);

            Directory.CreateDirectory(imageDir);
            Directory.CreateDirectory(annotationDir);

            var trainingList = new List<string>();

            Console.WriteLine($"🧩 Creating training batch: {batchId}");
            Console.WriteLine($"📂 Output folder: {baseDir}");

            for (int i = 0; i < numSamples; i++)
            {
                string imageName = $"img_{i:D4}.png";
                string imagePath = Path.Combine(imageDir, imageName);
                string annotationPath = Path.Combine(annotationDir, $"img_{i:D4}.json");

                var annotations = GenerateImageWithCircles(imagePath);
                SaveAnnotations(annotationPath, annotations, imageName);
                trainingList.Add(imageName);
            }

            // Label map
            string labelMapPath = Path.Combine(baseDir, AIConfig.LabelMapName);
            File.WriteAllText(labelMapPath, "item {\n  id: 1\n  name: 'circle'\n}");

            // Training list
            string trainListPath = Path.Combine(baseDir, AIConfig.TrainListName);
            File.WriteAllLines(trainListPath, trainingList);

            Console.WriteLine($"✅ Generated {numSamples} synthetic training images in batch {batchId}");

            return true;
        }

        /// <summary>
        /// Creates one synthetic image with non-overlapping circles.
        /// </summary>
        private static List<CircleAnnotation> GenerateImageWithCircles(string savePath)
        {
            var annotations = new List<CircleAnnotation>();

            using (var bmp = new Bitmap(AIConfig.ImageSize.Width, AIConfig.ImageSize.Height))
            using (var g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.White);

                int numCircles = rnd.Next(AIConfig.MinCircles, AIConfig.MaxCircles + 1);
                int maxTries = 1000; // prevent infinite loops
                int attempts = 0;

                while (annotations.Count < numCircles && attempts < maxTries)
                {
                    attempts++;

                    int radius = rnd.Next(5, 20);
                    int centerX = rnd.Next(radius, AIConfig.ImageSize.Width - radius);
                    int centerY = rnd.Next(radius, AIConfig.ImageSize.Height - radius);

                    var newCircle = new CircleAnnotation
                    {
                        Label = "circle",
                        XMin = centerX - radius,
                        YMin = centerY - radius,
                        XMax = centerX + radius,
                        YMax = centerY + radius
                    };

                    if (!OverlapsOrContains(newCircle, annotations))
                    {
                        var color = Color.FromArgb(
                            rnd.Next(80, 180),
                            rnd.Next(80, 200),
                            rnd.Next(80, 180));

                        using (var pen = new Pen(color, rnd.Next(2, 5)))
                        {
                            g.DrawEllipse(pen, centerX - radius, centerY - radius, radius * 2, radius * 2);
                        }

                        annotations.Add(newCircle);
                    }
                }

                bmp.Save(savePath, System.Drawing.Imaging.ImageFormat.Png);
            }

            return annotations;
        }

        /// <summary>
        /// Checks if the given circle overlaps or is inside another.
        /// </summary>
        private static bool OverlapsOrContains(CircleAnnotation candidate, List<CircleAnnotation> existing)
        {
            float cx1 = (candidate.XMin + candidate.XMax) / 2f;
            float cy1 = (candidate.YMin + candidate.YMax) / 2f;
            float r1 = (candidate.XMax - candidate.XMin) / 2f;

            foreach (var c in existing)
            {
                float cx2 = (c.XMin + c.XMax) / 2f;
                float cy2 = (c.YMin + c.YMax) / 2f;
                float r2 = (c.XMax - c.XMin) / 2f;

                float dx = cx1 - cx2;
                float dy = cy1 - cy2;
                float distance = MathF.Sqrt(dx * dx + dy * dy);

                // Overlap or containment check
                if (distance < (r1 + r2) || distance + Math.Min(r1, r2) < Math.Max(r1, r2))
                    return true;
            }

            return false;
        }

        private static void SaveAnnotations(string filePath, List<CircleAnnotation> circles, string imageName)
        {
            var annotationFile = new AnnotationFile
            {
                Image = imageName,
                Width = AIConfig.ImageSize.Width,
                Height = AIConfig.ImageSize.Height,
                Objects = circles
            };

            string json = JsonConvert.SerializeObject(annotationFile, Formatting.Indented);
            File.WriteAllText(filePath, json, Encoding.UTF8);
        }
    }
}
