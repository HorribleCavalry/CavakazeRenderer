using CavakazeRenderer.CrossPlatformAPIManager;
using CavakazeRenderer.Managers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CavakazeRenderer.ThreadStarts
{
    class RenderingStart
    {
        private int width;
        private int height;
        unsafe private void* imagePtr;
        unsafe public RenderingStart(ImageParam imageParam)
        {
            width = imageParam.width;
            height = imageParam.height;
            imagePtr = imageParam.imagePtr;
        }

        unsafe public void RenderThreadStart()
        {
            //UpdateTimer.Change(0, 16);
            //updateImageTicker.Change(Timeout.Infinite, 1);
            CudaAPI.StartRendering(width, height, imagePtr);
            //updateImageTicker.Change(0, 0);

            //UpdateTimer.Change(Timeout.Infinite, Timeout.Infinite);

        }
    }
}
