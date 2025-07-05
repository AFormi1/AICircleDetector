using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;

namespace AICircleDetector.AI
{
    public static class AIConfig
    {

        //Docs:
        //https://github.com/SciSharp/TensorFlow.NET
        //https://github.com/SciSharp/TensorFlow.NET/wiki/Using-GPU-with-Tensorflow.NET

        public static string TrainingFolderName { get; private set; } = "TrainingData";

        public static string ImageFolderName { get; private set; } = "images";
        public static string AnnotationsFolderName { get; private set; } = "annotations";
        public static string LabelMapName { get; private set; } = "label_map.pbtxt";
        public static string TrainListName { get; private set; } = "training.txt";
        public static string ValListName { get; private set; } = "validation.txt";
        public static string TrainingTF { get; private set; } = "training.tfrecord";
        public static string ValidationTF { get; private set; } = "validation.tfrecord";

        public static string TrainingModelFullURL { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TrainedModel");
        public static string TrainingModelName { get; set; } = "ObjectDetectionModel";

        public static int MaxCircles { get; private set; } = 10;
        public static int MinCircles { get; private set; } = 0;
        public static int ImageShape { get; private set; } = 28;
        public static Size ImageSize { get; private set; } = new Size(128, 128);
    }
}
