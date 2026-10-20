using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class AimShootViewer : MonoBehaviour
{
    public Shader surfaceShader;
    public Shader lineShader;
    [Serializable] public class Frame {
        public float timeMs, cameraAz, cameraEl, targetAz, targetEl, gazeX, gazeY;
    }
    [Serializable] public class Packet {
        public int episode; public Frame[] frames; public bool done, truncated, hit;
        public float timeMs, radius, errorDeg; public string error;
    }
    [Serializable] public class ResetRequest {
        public float[] parameters; public float targetSpeed, targetRadius; public int seed;
    }
    const string Endpoint = "http://127.0.0.1:8765";
    Camera eye;
    Transform target;
    Material targetMaterial;
    LineRenderer aimRay;
    Frame current = new Frame();
    Packet packet;
    bool running, paused, showGaze = true;
    float playbackSpeed = 1, trialSpeed = 12, trialRadius = 1.5f;
    float[] parameters = {.20f,.08f,.04f,.04f,40,12,40,45};
    string[] labels = {"Motor noise", "Position noise", "Speed noise", "Clock noise",
        "Hit reward", "Miss penalty", "Hit reward decay", "Miss penalty decay"};
    string status = "Ready. Press START AI to connect.";
    int seed = 1, shots, hits;
    float flashUntil;
    Vector2 scroll;

    void Start() {
        Application.runInBackground = true;
        Application.targetFrameRate = 60;
        QualitySettings.vSyncCount = 0;
        eye = new GameObject("AI first-person camera").AddComponent<Camera>();
        eye.clearFlags = CameraClearFlags.SolidColor;
        eye.backgroundColor = new Color(.025f,.04f,.07f);
        eye.fieldOfView = 70.533f; eye.nearClipPlane = .05f; eye.farClipPlane = 150;
        // Keep the original 16:9 projection, even when the application is resized.
        eye.aspect = 16f/9f;
        RenderSettings.ambientLight = new Color(.65f,.7f,.8f);
        var light = new GameObject("Key light").AddComponent<Light>();
        light.type = LightType.Directional; light.intensity = 1.1f;
        light.transform.rotation = Quaternion.Euler(40,-30,0);
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.name = "AI target"; target = sphere.transform;
        targetMaterial = new Material(surfaceShader);
        targetMaterial.color = new Color(.1f,.85f,.95f);
        sphere.GetComponent<Renderer>().material = targetMaterial;
        for(int i=-8;i<=8;i++) {
            Line(new Vector3(i*3,-7,-10),new Vector3(i*3,-7,60));
            Line(new Vector3(-24,-7,i*4+25),new Vector3(24,-7,i*4+25));
        }
        aimRay = new GameObject("Shot trace").AddComponent<LineRenderer>();
        aimRay.material = new Material(lineShader);
        aimRay.startWidth=.025f; aimRay.endWidth=.01f; aimRay.positionCount=2;
        aimRay.startColor=aimRay.endColor=Color.yellow; aimRay.enabled=false;
        Apply(new Frame {targetAz=20, targetEl=8}, trialRadius);
    }

    void Line(Vector3 a, Vector3 b) {
        var line = new GameObject("Arena grid").AddComponent<LineRenderer>();
        line.material = new Material(lineShader);
        line.startColor=line.endColor=new Color(.12f,.21f,.30f);
        line.startWidth=line.endWidth=.018f; line.positionCount=2;
        line.SetPosition(0,a); line.SetPosition(1,b);
    }

    // Source coordinates: X forward, Y up, Z right. Unity: Z forward, Y up, X right.
    public static Vector3 Direction(float az, float el) {
        float a=az*Mathf.Deg2Rad, e=el*Mathf.Deg2Rad;
        return new Vector3(Mathf.Sin(a)*Mathf.Cos(e),Mathf.Sin(e),Mathf.Cos(a)*Mathf.Cos(e));
    }

    void Apply(Frame f, float radius) {
        current=f;
        eye.transform.rotation=Quaternion.LookRotation(Direction(f.cameraAz,f.cameraEl),Vector3.up);
        target.position=Direction(f.targetAz,f.targetEl)*20;
        target.localScale=Vector3.one*(40*Mathf.Sin(radius*Mathf.Deg2Rad));
    }

    IEnumerator Request(string path, string json, Action<Packet> received) {
        using(var req=new UnityWebRequest(Endpoint+path,"POST")) {
            req.uploadHandler=new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
            req.downloadHandler=new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type","application/json"); req.timeout=15;
            yield return req.SendWebRequest();
            if(req.result!=UnityWebRequest.Result.Success) {
                status="Connection failed. Start the AI bridge. "+req.error;
                Debug.LogError(status+" "+req.downloadHandler.text); running=false; yield break;
            }
            var data=JsonUtility.FromJson<Packet>(req.downloadHandler.text);
            if(data==null || !string.IsNullOrEmpty(data.error) || data.frames==null) {
                status="AI bridge returned an invalid response."; running=false; yield break;
            }
            received(data);
        }
    }

    IEnumerator PlayPacket(Packet p) {
        var frames=p.frames;
        if(frames.Length==0) yield break;
        int index=0; float clock=frames[0].timeMs;
        while(clock < frames[frames.Length-1].timeMs && running) {
            if(!paused) clock+=Time.unscaledDeltaTime*1000*playbackSpeed;
            while(index+1<frames.Length && frames[index+1].timeMs<=clock) index++;
            Frame a=frames[index], b=frames[Mathf.Min(index+1,frames.Length-1)];
            float t=b.timeMs>a.timeMs?Mathf.Clamp01((clock-a.timeMs)/(b.timeMs-a.timeMs)):1;
            Apply(new Frame {timeMs=clock,
                cameraAz=Mathf.LerpAngle(a.cameraAz,b.cameraAz,t), cameraEl=Mathf.Lerp(a.cameraEl,b.cameraEl,t),
                targetAz=Mathf.LerpAngle(a.targetAz,b.targetAz,t), targetEl=Mathf.Lerp(a.targetEl,b.targetEl,t),
                gazeX=Mathf.Lerp(a.gazeX,b.gazeX,t), gazeY=Mathf.Lerp(a.gazeY,b.gazeY,t)},p.radius);
            yield return null;
        }
        Apply(frames[frames.Length-1],p.radius);
    }

    IEnumerator Run() {
        running=true; paused=false;
        do {
            status="Connecting to trained AI..."; packet=null;
            var request=new ResetRequest {parameters=parameters,targetSpeed=trialSpeed,targetRadius=trialRadius,seed=seed++};
            yield return Request("/reset",JsonUtility.ToJson(request),p=>packet=p);
            if(!running || packet==null) break;
            targetMaterial.color=new Color(.1f,.85f,.95f);
            yield return PlayPacket(packet);
            while(running && !packet.done) {
                while(paused && running) yield return null;
                if(!running) break;
                status="LIVE AI / decision interval 100 ms";
                yield return Request("/step","{}",p=>packet=p);
                if(!running) break;
                yield return PlayPacket(packet);
            }
            if(!running) break;
            shots++; if(packet.hit) hits++;
            status=packet.truncated?"TIMEOUT / target escaped":(packet.hit?"HIT":"MISS");
            targetMaterial.color=packet.hit?Color.green:Color.red;
            flashUntil=Time.unscaledTime+.18f;
            if(!packet.truncated) {
                aimRay.SetPosition(0,eye.transform.position+eye.transform.forward*.3f);
                aimRay.SetPosition(1,eye.transform.forward*22); aimRay.enabled=true;
            }
            yield return new WaitForSecondsRealtime(.65f);
            aimRay.enabled=false;
        } while(running);
        running=false;
    }

    void OnGUI() {
        if(eye==null) return;
        // Letterbox: maintain exact model projection and overlay coordinates.
        float w=Screen.width, h=Screen.height;
        float viewW=Mathf.Min(w,h*16f/9f), viewH=viewW*9f/16f;
        float ox=(w-viewW)/2, oy=(h-viewH)/2;
        eye.rect=new Rect(ox/w,oy/h,viewW/w,viewH/h);
        GUI.color=Time.unscaledTime<flashUntil?Color.yellow:Color.white;
        GUI.Label(new Rect(w/2-9,h/2-16,30,35),"+");
        if(showGaze) {
            GUI.color=new Color(1,.65f,.15f);
            GUI.Label(new Rect(ox+viewW*(.5f+current.gazeX/531.3f)-8,
                oy+viewH*(.5f-current.gazeY/298.8f)-12,70,30),"O gaze");
        }
        GUI.color=Color.white;
        GUILayout.BeginArea(new Rect(14,14,300,Screen.height-28),GUI.skin.box);
        scroll=GUILayout.BeginScrollView(scroll);
        GUILayout.Label("AIM + SHOOT | TRAINED AI");
        GUILayout.Label(status);
        GUILayout.Label("Episode "+(packet==null?0:packet.episode)+"   "+current.timeMs.ToString("F0")+" ms");
        GUILayout.Label("Hits "+hits+" / "+shots);
        if(packet!=null && packet.done) GUILayout.Label("Shot error "+packet.errorDeg.ToString("F2")+" deg");
        GUI.enabled=!running;
        if(GUILayout.Button("START AI / NEXT TRIAL",GUILayout.Height(32))) StartCoroutine(Run());
        GUI.enabled=true;
        if(running && GUILayout.Button(paused?"RESUME":"PAUSE")) paused=!paused;
        if(running && GUILayout.Button("STOP")) {running=false; StopAllCoroutines(); aimRay.enabled=false;}
        GUILayout.Label("Single shot -> reset -> next trial (auto)");
        showGaze=GUILayout.Toggle(showGaze,"Show AI gaze");
        GUILayout.Label("Playback speed "+playbackSpeed.ToString("F2")+"x");
        playbackSpeed=GUILayout.HorizontalSlider(playbackSpeed,.1f,2);
        GUILayout.Space(10); GUILayout.Label("NEXT TRIAL SETTINGS");
        GUILayout.Label("Target speed "+trialSpeed.ToString("F1")+" deg/s");
        trialSpeed=GUILayout.HorizontalSlider(trialSpeed,0,40);
        GUILayout.Label("Target radius "+trialRadius.ToString("F2")+" deg");
        trialRadius=GUILayout.HorizontalSlider(trialRadius,.3f,3);
        for(int i=0;i<8;i++) {
            GUILayout.Label(labels[i]+"  "+parameters[i].ToString("F2"));
            parameters[i]=GUILayout.HorizontalSlider(parameters[i],i<4?0:(i<6?1:5),i<4?.5f:(i<6?64:95));
        }
        GUILayout.Space(10);
        GUILayout.Label("Cyan: target | Orange: gaze\nWhite: crosshair | Yellow: shot\nSettings apply on next trial.\nMotion interpolated between model samples.");
        GUILayout.EndScrollView(); GUILayout.EndArea();
    }
}
