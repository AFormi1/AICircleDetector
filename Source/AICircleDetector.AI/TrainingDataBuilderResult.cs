using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AICircleDetector.AI
{
    public class TrainingDataBuilderResult
    {
        public bool Success { get; set; }
        public string Message { get; set; } = string.Empty;
        public string OutputDirectory { get; set; } = string.Empty;
    }
}
