using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;
using CavakazeRenderer.IDGenerator;
using CavakazeRenderer.CrossPlatformAPIManager;
using System.Windows.Forms;

namespace CavakazeRenderer.Managers
{
    unsafe struct ImageParam
    {
        public int width;
        public int height;
        public void* imagePtr;
    }
    class RenderManager: Manager
    {
        ThreadStarts.RenderingStart renderThreadStart;
        private Dictionary<ulong, Thread> threadPool;

        private IDGenerator.IDGenerator IDGenerator;

        private ulong renderThreadID;

        public RenderManager(ImageParam imageParam, ref PictureBox _RenderImage)
        {
            Initialize(imageParam, ref _RenderImage);
        }

        public void Initialize(ImageParam imageParam, ref PictureBox _RenderImage)
        {

            threadPool = new Dictionary<ulong, Thread>();
            IDGenerator = new OrderedIDGenerator();
            renderThreadID = IDGenerator.AllocateID();
            renderThreadStart = new ThreadStarts.RenderingStart(imageParam);
            threadPool.Add(renderThreadID, new Thread(renderThreadStart.RenderThreadStart));
        }

        unsafe public void StartRendering()
        {
            Thread renderThread = threadPool[renderThreadID];
            switch (renderThread.ThreadState)
            {
                case ThreadState.Running:
                    break;
                case ThreadState.StopRequested:
                    break;
                case ThreadState.SuspendRequested:
                    break;
                case ThreadState.Background:
                    break;
                case ThreadState.Unstarted:
                    renderThread.Start();
                    break;
                case ThreadState.Stopped:
                    break;
                case ThreadState.WaitSleepJoin:
                    break;
                case ThreadState.Suspended:
                    break;
                case ThreadState.AbortRequested:
                    break;
                case ThreadState.Aborted:
                    break;
                default:
                    break;
            }
        }

        public void Release()
        {

        }
    }
}
