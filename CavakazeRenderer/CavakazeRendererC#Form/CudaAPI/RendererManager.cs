using System;
using System.Runtime.InteropServices;


namespace CavakazeRenderer.CudaAPI
{
    class RenderManager
    {
        public RenderManager rendererManager;

        [DllImport("CudaAPI", EntryPoint = "StartRendering", CallingConvention = CallingConvention.Cdecl)]
        extern public static void StartRendering();
    }
}
