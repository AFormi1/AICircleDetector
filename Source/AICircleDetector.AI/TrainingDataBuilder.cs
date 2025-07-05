using Force.Crc32;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace AICircleDetector.AI
{
    public static class TrainingDataBuilder
    {
        private static readonly Random _rand = new();

        public static async Task<bool> CreateTrainingData(int imageCount)
        {
            try
            {
                string _currentGUID = Guid.NewGuid().ToString();

                Console.WriteLine($"CreateTrainingData {_currentGUID} has started, please wait ...");

                string _currentSessionDir = Path.Combine(Environment.CurrentDirectory, AIConfig.TrainingFolderName, _currentGUID);
                string _currentImageDir = Path.Combine(_currentSessionDir, AIConfig.ImageFolderName);
                string _currentAnnotationDir = Path.Combine(_currentSessionDir, AIConfig.AnnotationsFolderName);

                string labelMapPath = Path.Combine(_currentSessionDir, AIConfig.LabelMapName);

                string trainListPath = Path.Combine(_currentSessionDir, AIConfig.TrainListName);
                string valListPath = Path.Combine(_currentSessionDir, AIConfig.ValListName);

                string trainTfPath = Path.Combine(_currentSessionDir, AIConfig.TrainingTF);
                string valTfPath = Path.Combine(_currentSessionDir, AIConfig.ValidationTF);


                Directory.CreateDirectory(_currentImageDir!);
                Directory.CreateDirectory(_currentAnnotationDir!);

                for (int i = 0; i < imageCount; i++)
                {
                    int ringCount = _rand.Next(AIConfig.MinCircles, AIConfig.MaxCircles + 1);
                    string fileName = $"log_{i:D3}.png";
                    string imagePath = Path.Combine(_currentImageDir!, fileName);

                    var circles = GenerateRingImage(imagePath, ringCount);
                    SaveAnnotationXml(_currentAnnotationDir, fileName, circles);
                }

                CreateTrainValFiles(imageCount, trainListPath, valListPath, ".png", 0.2);

                CreateLabelMap(labelMapPath!);
                var classMap = ParseLabelMap(labelMapPath);

                CreateSerializedTFRecord(trainListPath, _currentImageDir, _currentAnnotationDir, classMap, trainTfPath);
                CreateSerializedTFRecord(valListPath, _currentImageDir, _currentAnnotationDir, classMap, valTfPath);

                Console.WriteLine($"CreateTrainingData {_currentGUID} finished!");

                return true;
            }
            catch (Exception ex)
            {
                return false;
            }
        }

        public static void CreateSerializedTFRecord(
            string listFilePath,
            string imageDir,
            string annotationDir,
            Dictionary<string, int> labelMap,
            string outputTFRecordPath)
        {
            using var outputStream = File.Create(outputTFRecordPath);

            foreach (var imageId in File.ReadAllLines(listFilePath))
            {
                string imagePath = Path.Combine(imageDir, imageId + ".png");
                string annotationPath = Path.Combine(annotationDir, imageId + ".xml");

                if (!File.Exists(imagePath) || !File.Exists(annotationPath))
                    continue;

                var imageBytes = File.ReadAllBytes(imagePath);
                var (xmins, xmaxs, ymins, ymaxs, labelsText, labelsIdx) = TfFeatureHelper.ParseAnnotation(annotationPath, labelMap);

                var features = new Features
                {
                    feature = new Dictionary<string, Feature>
                    {
                        ["image/encoded"] = TfFeatureHelper.Bytes(imageBytes),
                        ["image/filename"] = TfFeatureHelper.BytesList(new[] { imageId + ".png" }),
                        ["image/format"] = TfFeatureHelper.BytesList(new[] { "png" }),

                        ["image/object/bbox/xmin"] = TfFeatureHelper.FloatList(xmins),
                        ["image/object/bbox/xmax"] = TfFeatureHelper.FloatList(xmaxs),
                        ["image/object/bbox/ymin"] = TfFeatureHelper.FloatList(ymins),
                        ["image/object/bbox/ymax"] = TfFeatureHelper.FloatList(ymaxs),
                        ["image/object/class/text"] = TfFeatureHelper.BytesList(labelsText),
                        ["image/object/class/label"] = TfFeatureHelper.Int64List(labelsIdx),
                    }
                };

                var example = new Example { Features = features };

                // Serialize and write the Example
                using var ms = new MemoryStream();
                ProtoBuf.Serializer.Serialize(ms, example);

                var data = ms.ToArray();
                var lengthBytes = BitConverter.GetBytes((ulong)data.Length);
                outputStream.Write(lengthBytes);     // Write length
                outputStream.Write(data);            // Write serialized example
            }
        }


        private static uint Crc32C(byte[] data)
        {
            return Crc32CAlgorithm.Compute(data);
        }

        private static Dictionary<string, int> ParseLabelMap(string labelMapPath)
        {
            var lines = File.ReadAllLines(labelMapPath);
            var map = new Dictionary<string, int>();

            int? currentId = null;
            foreach (var line in lines)
            {
                if (line.Trim().StartsWith("id:"))
                    currentId = int.Parse(line.Split(':')[1].Trim());
                else if (line.Trim().StartsWith("name:") && currentId.HasValue)
                {
                    var name = line.Split(':')[1].Trim().Trim('\'', '"');
                    map[name] = currentId.Value;
                    currentId = null;
                }
            }
            return map;
        }


        private static void CreateLabelMap(string labelMapPath)
        {
            var labelMap = new StringBuilder();
            labelMap.AppendLine("item {");
            labelMap.AppendLine("  id: 1");
            labelMap.AppendLine("  name: 'circle'");
            labelMap.AppendLine("}");
            File.WriteAllText(labelMapPath, labelMap.ToString());
        }

        private static void CreateTrainValFiles(
            int imageCount,
            string trainListPath,
            string valListPath,
            string imageExtension = ".png",
            double valSplit = 0.2)
        {
            var allFiles = Enumerable.Range(0, imageCount)
                                      .Select(i => $"log_{i:D3}")
                                      .OrderBy(name => name) // Sort alphabetically/logically
                                      .ToList();

            int valCount = (int)(imageCount * valSplit);
            var valFiles = allFiles.Take(valCount).ToList();
            var trainFiles = allFiles.Skip(valCount).ToList();

            File.WriteAllLines(trainListPath, trainFiles);
            File.WriteAllLines(valListPath, valFiles);
        }



        private static void SaveAnnotationXml(string annotationDir, string imageFileName, List<(float x, float y, float r)> circles)
        {
            string fileNameWithoutExt = Path.GetFileNameWithoutExtension(imageFileName);
            string xmlPath = Path.Combine(annotationDir, $"{fileNameWithoutExt}.xml");

            var imageWidth = AIConfig.ImageSize.Width;
            var imageHeight = AIConfig.ImageSize.Height;

            var doc = new XDocument(
                new XElement("annotation",
                    new XElement("folder", "images"),
                    new XElement("filename", imageFileName),
                    new XElement("size",
                        new XElement("width", imageWidth),
                        new XElement("height", imageHeight),
                        new XElement("depth", 3)
                    ),
                    circles.Select(c => {
                        int xmin = (int)(c.x - c.r);
                        int ymin = (int)(c.y - c.r);
                        int xmax = (int)(c.x + c.r);
                        int ymax = (int)(c.y + c.r);

                        return new XElement("object",
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
                    })
                )
            );

            doc.Save(xmlPath);
        }


        private static List<(float x, float y, float r)> GenerateRingImage(string filePath, int ringCount)
        {
            const int maxAttempts = 1000;
            var placedCircles = new List<(float x, float y, float r)>();

            using SKBitmap bitmap = new SKBitmap(AIConfig.ImageSize.Width, AIConfig.ImageSize.Height);
            using SKCanvas canvas = new SKCanvas(bitmap);
            canvas.Clear(SKColors.White);

            int attempts = 0;
            while (placedCircles.Count < ringCount && attempts < maxAttempts)
            {
                float radius = _rand.Next(5, (AIConfig.ImageSize.Height + AIConfig.ImageSize.Width) / 2 / 4);
                float x = _rand.Next((int)radius, AIConfig.ImageSize.Width - (int)radius);
                float y = _rand.Next((int)radius, AIConfig.ImageSize.Height - (int)radius);

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

        private static SKColor RandomGreenBrown()
        {
            int r = _rand.Next(60, 160);
            int g = _rand.Next(80, 180);
            int b = _rand.Next(30, 100);
            return new SKColor((byte)r, (byte)g, (byte)b);
        }
    }
}
