using SkiaSharp;
using System.Drawing;
using System.Xml.Linq;

namespace AICircleDetector.AI
{
    public static class TrainingDataBuilder
    {
        public  static string? _currentSessionDir;
        private static string? _currentImageDir;
        private static string? _currentCsvPath;
        private static string? _currentGUID;
        private static readonly Random _rand = new();

        public static TrainingDataBuilderResult CreateTrainingData(int imageCount = 20)
        {
            try
            {
                _currentGUID = Guid.NewGuid().ToString(); // DateTime.Now.ToString("yyyyMMdd_HHmmss");
                _currentSessionDir = Path.Combine(Environment.CurrentDirectory, "TrainingData", _currentGUID);
                _currentImageDir = Path.Combine(_currentSessionDir, "images");
                _currentCsvPath = Path.Combine(_currentSessionDir, "labels.csv");

                Directory.CreateDirectory(_currentImageDir!);
                File.WriteAllText(_currentCsvPath!, "ImagePath;CircleCount;Session\n");

                for (int i = 0; i < imageCount; i++)
                {
                    int ringCount = _rand.Next(AIConfig.MinCircles, AIConfig.MaxCircles + 1);
                    string fileName = $"log_{i:D3}.png";
                    string filePath = Path.Combine(_currentImageDir!, fileName);

                    var circles = GenerateRingImage(filePath, ringCount);
                    AddXmlAnnotation(fileName, filePath, circles);

                    AddCsvEntry(fileName, ringCount);
                }

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

        private static void AddXmlAnnotation(string fileName, string filePath, List<(float x, float y, float r)> circles)
        {
            int imageWidth = AIConfig.ImageSize;
            int imageHeight = AIConfig.ImageSize;

            var annotation = new XElement("annotation",
                new XElement("folder", "images"),
                new XElement("filename", fileName),
                new XElement("path", filePath),
                new XElement("source", new XElement("database", "Synthetic")),
                new XElement("size",
                    new XElement("width", imageWidth),
                    new XElement("height", imageHeight),
                    new XElement("depth", 3)
                ),
                new XElement("segmented", 0)
            );

            foreach (var (x, y, r) in circles)
            {
                int xmin = (int)Math.Max(x - r, 0);
                int ymin = (int)Math.Max(y - r, 0);
                int xmax = (int)Math.Min(x + r, imageWidth);
                int ymax = (int)Math.Min(y + r, imageHeight);

                var obj = new XElement("object",
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
                );
                annotation.Add(obj);
            }

            string xmlFilePath = Path.Combine(_currentImageDir!, Path.GetFileNameWithoutExtension(fileName) + ".xml");
            annotation.Save(xmlFilePath);
        }



        private static List<(float x, float y, float r)> GenerateRingImage(string filePath, int ringCount)
        {
            const int maxAttempts = 1000;
            using SKBitmap bitmap = new SKBitmap(AIConfig.ImageSize, AIConfig.ImageSize);
            using SKCanvas canvas = new SKCanvas(bitmap);
            canvas.Clear(SKColors.White);

            List<(float x, float y, float r)> placedCircles = new();
            int attempts = 0;

            while (placedCircles.Count < ringCount && attempts < maxAttempts)
            {
                float radius = _rand.Next(5, AIConfig.ImageSize / 4);
                float x = _rand.Next((int)radius, AIConfig.ImageSize - (int)radius);
                float y = _rand.Next((int)radius, AIConfig.ImageSize - (int)radius);

                bool collides = placedCircles.Any(c =>
                {
                    float dx = c.x - x, dy = c.y - y;
                    float distSq = dx * dx + dy * dy;
                    return distSq < MathF.Pow(c.r + radius + 1, 2);
                });

                if (collides)
                {
                    attempts++;
                    continue;
                }

                placedCircles.Add((x, y, radius));

                float thickness = _rand.Next(1, 5);
                using SKPaint paint = new SKPaint
                {
                    Style = SKPaintStyle.Stroke,
                    Color = RandomGreenBrown(),
                    StrokeWidth = thickness,
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



        private static SKColor RandomGreenBrown()
        {
            int r = _rand.Next(60, 160);
            int g = _rand.Next(80, 180);
            int b = _rand.Next(30, 100);
            return new SKColor((byte)r, (byte)g, (byte)b);
        }

        private static void AddCsvEntry(string fileName, int ringCount)
        {
            string relativePath = Path.Combine("images", fileName);
            string csvLine = $"{relativePath};{ringCount};{_currentGUID}\n";
            File.AppendAllText(_currentCsvPath!, csvLine);
        }

        public static string GetCurrentDatasetPath() => _currentSessionDir!;
    }
}