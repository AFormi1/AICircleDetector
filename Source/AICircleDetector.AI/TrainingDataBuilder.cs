using OneOf.Types;
using SkiaSharp;
using System;
using System.Drawing;
using System.Xml.Linq;

namespace AICircleDetector.AI
{
    public static class TrainingDataBuilder
    {
        public static string? _currentSessionDir;
        private static string? _currentImageDir;
        private static string? _currentAnnotationDir;
        private static string? _currentGUID;
        private static readonly Random _rand = new();

        public static TrainingDataBuilderResult CreateTrainingData(int imageCount = 20)
        {
            try
            {
                _currentGUID = Guid.NewGuid().ToString();
                _currentSessionDir = Path.Combine(Environment.CurrentDirectory, "TrainingData", _currentGUID);
                _currentImageDir = Path.Combine(_currentSessionDir, "images");
                _currentAnnotationDir = Path.Combine(_currentSessionDir, "annotations");

                Directory.CreateDirectory(_currentImageDir!);
                Directory.CreateDirectory(_currentAnnotationDir!);

                for (int i = 0; i < imageCount; i++)
                {
                    int ringCount = _rand.Next(AIConfig.MinCircles, AIConfig.MaxCircles + 1);
                    string fileName = $"log_{i:D3}.png";
                    string imagePath = Path.Combine(_currentImageDir!, fileName);

                    var circles = GenerateRingImage(imagePath, ringCount);
                    SaveAnnotationXml(fileName, circles);                  
                }

                CreateLabelMap(_currentSessionDir!);

                CreateTrainValFiles(imageCount, _currentSessionDir!);

                return new TrainingDataBuilderResult
                {
                    Success = true,
                    Message = $"Successfully generated {imageCount} training images.",
                    OutputDirectory = _currentSessionDir!
                };
            }
            catch (Exception ex)
            {
                return new TrainingDataBuilderResult
                {
                    Success = false,
                    Message = $"Error during generation: {ex.Message}",
                    OutputDirectory = string.Empty
                };
            }
        }

        private static List<(float x, float y, float r)> GenerateRingImage(string filePath, int ringCount)
        {
            const int maxAttempts = 1000;
            var placedCircles = new List<(float x, float y, float r)>();

            using SKBitmap bitmap = new SKBitmap(AIConfig.ImageSize, AIConfig.ImageSize);
            using SKCanvas canvas = new SKCanvas(bitmap);
            canvas.Clear(SKColors.White);

            int attempts = 0;
            while (placedCircles.Count < ringCount && attempts < maxAttempts)
            {
                float radius = _rand.Next(5, AIConfig.ImageSize / 4);
                float x = _rand.Next((int)radius, AIConfig.ImageSize - (int)radius);
                float y = _rand.Next((int)radius, AIConfig.ImageSize - (int)radius);

                bool collides = placedCircles.Any(c =>
                {
                    float dx = c.x - x;
                    float dy = c.y - y;
                    float distSq = dx * dx + dy * dy;
                    float minDist = c.r + radius + 1;
                    return distSq < minDist * minDist;
                });

                if (collides)
                {
                    attempts++;
                    continue;
                }

                placedCircles.Add((x, y, radius));

                using SKPaint paint = new SKPaint
                {
                    Style = SKPaintStyle.Stroke,
                    Color = RandomGreenBrown(),
                    StrokeWidth = _rand.Next(1, 5),
                    IsAntialias = true
                };

                canvas.DrawCircle(x, y, radius, paint);
            }

            using SKImage image = SKImage.FromBitmap(bitmap);
            using SKData data = image.Encode(SKEncodedImageFormat.Png, 100);
            using FileStream stream = File.OpenWrite(filePath);
            data.SaveTo(stream);

            return placedCircles;
        }

        private static void SaveAnnotationXml(string imageFileName, List<(float x, float y, float r)> circles)
        {
            string imagePath = Path.Combine("images", imageFileName);
            string annotationPath = Path.Combine(_currentAnnotationDir!, Path.ChangeExtension(imageFileName, ".xml"));

            var annotation = new XElement("annotation",
                new XElement("folder", "images"),
                new XElement("filename", imageFileName),
                new XElement("path", imagePath),
                new XElement("source",
                    new XElement("database", "Unknown")),
                new XElement("size",
                    new XElement("width", AIConfig.ImageSize),
                    new XElement("height", AIConfig.ImageSize),
                    new XElement("depth", 3)),
                new XElement("segmented", 0)
            );

            foreach (var (x, y, r) in circles)
            {
                int xmin = (int)(x - r);
                int ymin = (int)(y - r);
                int xmax = (int)(x + r);
                int ymax = (int)(y + r);

                annotation.Add(
                    new XElement("object",
                        new XElement("name", "circle"),
                        new XElement("pose", "Unspecified"),
                        new XElement("truncated", 0),
                        new XElement("difficult", 0),
                        new XElement("bndbox",
                            new XElement("xmin", xmin),
                            new XElement("ymin", ymin),
                            new XElement("xmax", xmax),
                            new XElement("ymax", ymax)
                        )
                    )
                );
            }

            XDocument xmlDoc = new(annotation);
            xmlDoc.Save(annotationPath);
        }

        private static SKColor RandomGreenBrown()
        {
            int r = _rand.Next(60, 160);
            int g = _rand.Next(80, 180);
            int b = _rand.Next(30, 100);
            return new SKColor((byte)r, (byte)g, (byte)b);
        }

        private static void CreateLabelMap(string outputDir)
        {
            string labelMapPath = Path.Combine(outputDir, "label_map.pbtxt");
            var content = "item {\n  id: 1\n  name: 'circle'\n}\n";
            File.WriteAllText(labelMapPath, content);
        }

        private static void CreateTrainValFiles(int imageCount, string outputDir, double validationSplit = 0.2)
        {
            string path = Path.Combine(outputDir, "trainval.txt");
            var allFilenames = Enumerable.Range(0, imageCount)
                                         .Select(i => $"log_{i:D3}")
                                         .ToList();

            // Shuffle the filenames to ensure randomness
            var shuffledFilenames = allFilenames.OrderBy(x => _rand.Next()).ToList();

            // Split the shuffled list into train and val based on the split ratio (80% train, 20% val)
            int valCount = (int)(imageCount * validationSplit);  // 20% for validation
            var valFilenames = shuffledFilenames.Take(valCount).ToList();
            var trainFilenames = shuffledFilenames.Skip(valCount).ToList();

            // Write to trainval.txt (all filenames)
            File.WriteAllLines(path, allFilenames);

            // Write to train.txt (80% for training)
            string trainPath = Path.Combine(outputDir, "train.txt");
            File.WriteAllLines(trainPath, trainFilenames);

            // Write to val.txt (20% for validation)
            string valPath = Path.Combine(outputDir, "val.txt");
            File.WriteAllLines(valPath, valFilenames);
        }


    }
}
