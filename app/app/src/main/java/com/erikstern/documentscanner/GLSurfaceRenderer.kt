package com.erikstern.documentscanner

import android.content.Context
import android.content.res.AssetManager
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.view.Surface
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class GLSurfaceRenderer : GLSurfaceView.Renderer {
    lateinit var context: Context

    var surfaceTextureId = -1
    lateinit var surfaceTexture: SurfaceTexture
    lateinit var surface: Surface
    var nativeContext: Long = 0

    val lock = Any()
    var frameAvailable = false

    private external fun nativeCreate()
    private external fun nativeDestroy()
    private external fun nativePreInit() : IntArray
    private external fun nativeInit(assetManager: AssetManager, surface: Surface, preview_width: Int, preview_height: Int)

    private external fun nativeRender()

    fun setSurfaceView(surfaceView: GLSurfaceView) {
        surfaceView.setEGLContextClientVersion(3)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        context = surfaceView.context

        nativeCreate()
    }

    fun destroyRender() {
        nativeDestroy()
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        surfaceTextureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, surfaceTextureId)

        surfaceTexture = SurfaceTexture(surfaceTextureId)
        surfaceTexture.setOnFrameAvailableListener {
            synchronized(lock) {
                frameAvailable = true
            }
        }

        surface = Surface(surfaceTexture)
        val dimens = nativePreInit()
        surfaceTexture.setDefaultBufferSize(dimens[0], dimens[1])
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        nativeInit(context.assets, surface, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        synchronized(lock) {
            if (frameAvailable) {
                surfaceTexture.updateTexImage()
                frameAvailable = false
            }
        }

        nativeRender()
    }
}