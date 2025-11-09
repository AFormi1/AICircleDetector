using System;
using System.Collections.Generic;

namespace AICircleDetector.AI
{
    [Serializable]
    public class AnnotationFile
    {
        public string Image { get; set; } = string.Empty;
        public int Width { get; set; }
        public int Height { get; set; }
        public List<CircleAnnotation> Objects { get; set; } = new();
    }
}
