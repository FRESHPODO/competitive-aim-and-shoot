using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class DualAimShootViewer : MonoBehaviour
{
    public Shader surfaceShader, lineShader;
    [Serializable] public class Frame { public float timeMs,cameraAz,cameraEl,targetAz,targetEl,gazeX,gazeY; public bool fireCommitted,cooldown; }
    [Serializable] public class ActionState { public float th,kc,tc,kg; }
    [Serializable] public class AgentPacket {
        public Frame[] frames; public bool done,hit,truncated,cooldown;
        public float timeMs,radius,errorDeg; public ActionState action;
    }
    [Serializable] public class DamageEvent {
        public int agent; public float timeMs,damage,normalizedError,scoreDelta,hpAfter;
        public float shotAz,shotEl,spreadAngle;
        public bool critical,hit,killed;
    }
    [Serializable] public class RoundPacket {
        public int round; public AgentPacket[] agents; public float timeMs,targetHp,maxHp,motionSpeed;
        public bool done,destroyed; public string error;
        public DamageEvent[] events; public float[] initialScores,scores;
        public int[] initialBullets,initialHits,bullets;
    }
    [Serializable] public class ScoreResetRequest { public int round; public float timeMs; }
    [Serializable] public class ResetRequest { public float[][] profiles; public float targetRadius; }
    // JsonUtility does not support jagged arrays: send this explicit wire type.
    [Serializable] public class Profile { public float[] values; }
    class AgentView {
        public Camera camera; public Transform body; public LineRenderer ray;
        public Frame frame=new Frame(); public AgentPacket packet;
        public int shots,hits,targetShots; public bool counted,criticalKill; public Vector2 scroll;
        public float score,lastDamage,targetDamage; public string shotStatus="";
        public LineRenderer bulletRay; public float shotAz,shotEl,shotDeviation,lastShotTime=-999;
    }
    readonly AgentView[] views={new AgentView(),new AgentView()};
    readonly Vector3[] origins={Vector3.zero,Vector3.zero};
    readonly Color[] colors={new Color(.2f,.75f,1),new Color(1,.55f,.25f)};
    readonly float[][] profiles={new float[]{.20f,.08f,.04f,.04f,40,12,40,45},new float[]{.20f,.08f,.04f,.04f,40,12,40,45}};
    readonly string[] labels={"Motor noise","Position noise","Speed noise","Clock noise","Hit reward","Miss penalty","Hit decay","Miss decay"};
    Camera observer; Transform target; RoundPacket round;
    bool running,paused,showGaze=true; float speed=.5f,radius=5f,clock;
    float targetHp=1000,minDamage=100,maxDamage=300,criticalRadius=.01f,visibleHp=1000;
    readonly float[] shotIntervals={150,150};
    readonly float[] shotSpreads={0,0};
    float targetSpeed=10,targetSpeedMin=0,targetSpeedMax=40;
    bool randomTargetSpawn=true,randomTargetSpeed=true;
    Vector2 targetScroll,controlScroll;
    float resultPause=1.5f;
    bool showingResult;
    string resultLogPath;
    bool scoreResetBusy,modelRequestBusy;
    int appliedEvents;
    bool exploded;
    Material targetMaterial;
    GameObject debris;
    string status="Ready: two synchronized agents / one shared target";
    const string Host="http://127.0.0.1:8765";
    Rect[] personal=new Rect[2];
    float SidePanelWidth { get { return Mathf.Clamp(Screen.width*.18f,150,300); } }
    float CenterPanelX { get { return SidePanelWidth+24; } }
    float CenterSpace { get { return Screen.width-SidePanelWidth*2-48; } }
    float ControlWidth { get { return Mathf.Max(120,Mathf.Min(340,CenterSpace*.47f)); } }
    float PanelHeight => Mathf.Max(100,Screen.height*.48f-24);
    Rect AgentPanel(int i) => new Rect(i==0?12:Screen.width-SidePanelWidth-12,12,SidePanelWidth,PanelHeight);
    Rect ControlPanel => new Rect(CenterPanelX,12,ControlWidth,Mathf.Max(45,(PanelHeight-10)*.48f));
    Rect TargetPanel => new Rect(CenterPanelX,ControlPanel.yMax+10,ControlWidth,Mathf.Max(45,PanelHeight-ControlPanel.height-10));
    Rect ScorePanel {
        get {
            float width=Mathf.Min(320,Mathf.Max(80,CenterSpace-ControlWidth-12));
            return new Rect(AgentPanel(1).x-12-width,12,width,Mathf.Min(260,PanelHeight));
        }
    }
    GUIStyle panelStyle,headingStyle;
    void BeginPanel(Rect rect,string title,ref Vector2 scroll) {
        GUILayout.BeginArea(rect,panelStyle);
        GUILayout.Label(title,headingStyle);
        scroll=GUILayout.BeginScrollView(scroll,false,true);
    }
    void EndPanel() { GUILayout.EndScrollView();GUILayout.EndArea(); }
    bool observerFocused,observerDragging,numericFocused;
    Vector3 previousMouse;
    float observerYaw,observerPitch;
    Font uiFont;

    void Start() {
        uiFont=Font.CreateDynamicFontFromOSFont("Malgun Gothic",13);
        resultLogPath=Path.Combine(Application.persistentDataPath,"target-results.csv");
        Application.runInBackground=true; Application.targetFrameRate=60;
        QualitySettings.vSyncCount=0;
        RenderSettings.ambientLight=new Color(.65f,.7f,.8f);
        var light=new GameObject("Arena light").AddComponent<Light>();
        light.type=LightType.Directional; light.intensity=1.1f;
        light.transform.rotation=Quaternion.Euler(35,-20,0);
        target=Ball("Shared moving target",new Vector3(0,2,20),Color.cyan,.5f).transform;
        targetMaterial=target.GetComponent<Renderer>().material;
        for(int i=-8;i<=8;i++) {
            MakeLine("Floor",new Vector3(i*3,-1.5f,-8),new Vector3(i*3,-1.5f,32),new Color(.12f,.2f,.3f),.02f);
            MakeLine("Floor",new Vector3(-24,-1.5f,i*3+12),new Vector3(24,-1.5f,i*3+12),new Color(.12f,.2f,.3f),.02f);
        }
        observer=NewCamera("Observer",new Color(.025f,.04f,.07f),0);
        observer.transform.position=new Vector3(0,10,-12);
        observer.transform.LookAt(new Vector3(0,0,10)); observer.fieldOfView=62;
        observerYaw=observer.transform.eulerAngles.y;
        observerPitch=observer.transform.eulerAngles.x;
        if(observerPitch>180)observerPitch-=360;
        for(int i=0;i<2;i++) {
            var v=views[i];
            v.camera=NewCamera("Agent "+(i+1)+" first person",new Color(.025f,.04f,.07f),i+1);
            v.camera.transform.position=origins[i]; v.camera.fieldOfView=70.533f;
            var body=GameObject.CreatePrimitive(PrimitiveType.Capsule);
            body.name="Agent "+(i+1); body.transform.position=origins[i]+Vector3.down*.7f;
            body.transform.localScale=new Vector3(.45f,.7f,.45f);
            var material=new Material(surfaceShader); material.color=colors[i];
            body.GetComponent<Renderer>().material=material; body.layer=8+i;
            // Both model cameras share the exact origin. One neutral rig represents them.
            body.GetComponent<Renderer>().enabled=i==0;
            if(i==0)material.color=new Color(.7f,.7f,.8f);
            v.body=body.transform; v.camera.cullingMask=~((1<<8) | (1<<9) | (1<<10));
            v.ray=MakeLine("Agent "+(i+1)+" aim",origins[i],target.position,colors[i],.015f);
            v.ray.gameObject.layer=10;
            Color bulletColor=colors[i];bulletColor.a=.4f;
            v.bulletRay=MakeLine("Agent "+(i+1)+" bullet",origins[i],target.position,bulletColor,.065f);
            v.bulletRay.gameObject.layer=10;v.bulletRay.enabled=false;
            v.frame=new Frame {targetAz=0,targetEl=5.71f};
        }
        SetRadius(radius);
        Layout();
    }

    Camera NewCamera(string name,Color background,int depth) {
        var camera=new GameObject(name).AddComponent<Camera>(); camera.backgroundColor=background;
        camera.clearFlags=CameraClearFlags.SolidColor; camera.depth=depth;
        camera.nearClipPlane=.05f; camera.farClipPlane=120; return camera;
    }
    GameObject Ball(string name,Vector3 position,Color color,float size) {
        var go=GameObject.CreatePrimitive(PrimitiveType.Sphere); go.name=name;
        go.transform.position=position; go.transform.localScale=Vector3.one*size;
        var mat=new Material(surfaceShader); mat.color=color; go.GetComponent<Renderer>().material=mat; return go;
    }
    LineRenderer MakeLine(string name,Vector3 a,Vector3 b,Color color,float width) {
        var line=new GameObject(name).AddComponent<LineRenderer>();
        line.material=new Material(lineShader); line.startColor=line.endColor=color;
        line.startWidth=line.endWidth=width; line.positionCount=2;
        line.SetPosition(0,a); line.SetPosition(1,b); return line;
    }
    void SetRadius(float value) { target.localScale=Vector3.one*(2*Mathf.Sqrt(404)*Mathf.Sin(value*Mathf.Deg2Rad)); }
    public static Vector3 Direction(float az,float el) {
        float a=az*Mathf.Deg2Rad,e=el*Mathf.Deg2Rad;
        return new Vector3(Mathf.Sin(a)*Mathf.Cos(e),Mathf.Sin(e),Mathf.Cos(a)*Mathf.Cos(e));
    }
    void Layout() {
        observer.rect=new Rect(0,.52f,1,.48f);
        for(int i=0;i<2;i++) {
            float slotWidth=Screen.width*.5f, slotHeight=Screen.height*.48f;
            float width=Mathf.Min(slotWidth,slotHeight*16f/9f), height=width*9f/16f;
            personal[i]=new Rect(i*slotWidth+(slotWidth-width)*.5f,Screen.height*.52f+(slotHeight-height)*.5f,width,height);
            views[i].camera.rect=new Rect(personal[i].x/Screen.width,1-(personal[i].y+height)/Screen.height,width/Screen.width,height/Screen.height);
            views[i].camera.aspect=16f/9;
        }
    }
    bool IsObserverUI(Vector2 point) {
        return AgentPanel(0).Contains(point) || AgentPanel(1).Contains(point) ||
            ControlPanel.Contains(point) || TargetPanel.Contains(point) || ScorePanel.Contains(point);
    }
    void Update() {
        Layout();
        Vector3 mouse=Input.mousePosition;
        Vector2 guiPoint=new Vector2(mouse.x,Screen.height-mouse.y);
        bool navigable=guiPoint.y>=0 && guiPoint.y<Screen.height*.48f && !IsObserverUI(guiPoint);
        if(Input.GetMouseButtonDown(0) || Input.GetMouseButtonDown(1)) {
            observerFocused=navigable;
            observerDragging=navigable;
            previousMouse=mouse;
        }
        if(!Input.GetMouseButton(0) && !Input.GetMouseButton(1))observerDragging=false;
        if(observerDragging) {
            Vector3 delta=mouse-previousMouse;
            observerYaw+=delta.x*.15f;
            observerPitch=Mathf.Clamp(observerPitch-delta.y*.15f,-85,85);
            observer.transform.rotation=Quaternion.Euler(observerPitch,observerYaw,0);
        }
        previousMouse=mouse;
        if(observerFocused && !numericFocused) {
            float horizontal=(Input.GetKey(KeyCode.D)?1:0)-(Input.GetKey(KeyCode.A)?1:0);
            float forward=(Input.GetKey(KeyCode.W)?1:0)-(Input.GetKey(KeyCode.S)?1:0);
            float vertical=(Input.GetKey(KeyCode.E)?1:0)-(Input.GetKey(KeyCode.Q)?1:0);
            Vector3 movement=observer.transform.right*horizontal+observer.transform.forward*forward+Vector3.up*vertical;
            float velocity=Input.GetKey(KeyCode.LeftShift)?20:8;
            observer.transform.position+=Vector3.ClampMagnitude(movement,1)*velocity*Time.unscaledDeltaTime;
        }
        if(!Application.isFocused){observerDragging=false;observerFocused=false;}
    }

    IEnumerator Request(string path,string json,Action<RoundPacket> callback) {
        modelRequestBusy=true;
        using(var req=new UnityWebRequest(Host+path,"POST")) {
            req.uploadHandler=new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
            req.downloadHandler=new DownloadHandlerBuffer(); req.SetRequestHeader("Content-Type","application/json"); req.timeout=20;
            yield return req.SendWebRequest();
            modelRequestBusy=false;
            if(req.result!=UnityWebRequest.Result.Success) { status="AI connection failed: "+req.error; Debug.LogError(status+req.downloadHandler.text); running=false; yield break; }
            var data=JsonUtility.FromJson<RoundPacket>(req.downloadHandler.text);
            if(data==null || data.agents==null || data.agents.Length!=2) {status="Bridge protocol mismatch. Restart bridge."; running=false; yield break;}
            callback(data);
        }
    }
    static Frame Sample(Frame[] frames,float time) {
        int k=0; while(k+1<frames.Length && frames[k+1].timeMs<=time) k++;
        var a=frames[k]; var b=frames[Mathf.Min(k+1,frames.Length-1)];
        float t=b.timeMs>a.timeMs?Mathf.Clamp01((time-a.timeMs)/(b.timeMs-a.timeMs)):1;
        return new Frame {timeMs=Mathf.Min(time,frames[frames.Length-1].timeMs),fireCommitted=a.fireCommitted,cooldown=a.cooldown,
            cameraAz=Mathf.LerpAngle(a.cameraAz,b.cameraAz,t),cameraEl=Mathf.Lerp(a.cameraEl,b.cameraEl,t),
            targetAz=Mathf.LerpAngle(a.targetAz,b.targetAz,t),targetEl=Mathf.LerpAngle(a.targetEl,b.targetEl,t),
            gazeX=Mathf.Lerp(a.gazeX,b.gazeX,t),gazeY=Mathf.Lerp(a.gazeY,b.gazeY,t)};
    }
    void ApplyAt(float time) {
        // Same target pose for both observers, sampled at the common clock.
        var targetFrame=Sample(round.agents[0].frames,time);
        target.position=Direction(targetFrame.targetAz,targetFrame.targetEl)*Mathf.Sqrt(404);
        visibleHp=round.maxHp;
        for(int k=0;k<round.events.Length;k++) {
            var e=round.events[k]; if(e.timeMs>time)break;
            visibleHp=e.hpAfter;
            if(k>=appliedEvents) {
                var shooter=views[e.agent];
                shooter.lastDamage=e.damage;
                if(e.critical && e.killed)shooter.criticalKill=true;
                shooter.shotStatus=e.critical?"CRITICAL":(e.hit?"HIT":"MISS");
                shooter.shotAz=e.shotAz;shooter.shotEl=e.shotEl;shooter.shotDeviation=e.spreadAngle;shooter.lastShotTime=e.timeMs;
                appliedEvents=k+1;
            }
        }
        bool destroyed=visibleHp<=0;
        target.gameObject.SetActive(!destroyed);
        targetMaterial.color=Color.Lerp(new Color(1,.2f,.1f),Color.cyan,visibleHp/round.maxHp);
        if(destroyed && !exploded) {
            exploded=true;
            debris=new GameObject("Destroyed target fragments");
            for(int j=0;j<12;j++) {
                float angle=j*Mathf.PI*2/12;
                var piece=Ball("Fragment",target.position+new Vector3(Mathf.Cos(angle),Mathf.Sin(angle),0)*.8f,new Color(1,.6f,.15f),.12f);
                piece.transform.SetParent(debris.transform);
            }
        }
        for(int i=0;i<2;i++) {
            var v=views[i]; v.packet=round.agents[i];
            v.score=round.initialScores[i];
            v.shots=round.initialBullets[i];v.hits=round.initialHits[i];v.targetShots=0;v.targetDamage=0;
            foreach(var e in round.events)if(e.agent==i && e.timeMs<=time){v.score+=e.scoreDelta;v.targetDamage+=e.damage;v.shots++;v.targetShots++;if(e.hit)v.hits++;}
            v.frame=Sample(v.packet.frames,time);
            var direction=Direction(v.frame.cameraAz,v.frame.cameraEl);
            v.camera.transform.rotation=Quaternion.LookRotation(direction,Vector3.up);
            v.body.rotation=Quaternion.Euler(0,v.frame.cameraAz,0);
            v.ray.startColor=v.ray.endColor=colors[i];
            v.ray.SetPosition(0,origins[i]); v.ray.SetPosition(1,origins[i]+direction*23);
            v.bulletRay.enabled=time>=v.lastShotTime && time-v.lastShotTime<180;
            v.bulletRay.SetPosition(0,origins[i]);v.bulletRay.SetPosition(1,origins[i]+Direction(v.shotAz,v.shotEl)*23);
        }
    }
    IEnumerator Advance() {
        while(clock<round.timeMs && running) {
            if(!paused && !scoreResetBusy) clock=Mathf.Min(round.timeMs,clock+Time.unscaledDeltaTime*1000*speed);
            ApplyAt(clock); yield return null;
        }
        ApplyAt(round.timeMs);
    }
    string ResetJson() {
        // The rest of the request is serialized normally; arrays are explicit JSON.
        string a=JsonUtility.ToJson(new Profile {values=profiles[0]});
        string b=JsonUtility.ToJson(new Profile {values=profiles[1]});
        a=a.Substring(a.IndexOf('[')); a=a.Substring(0,a.LastIndexOf(']')+1);
        b=b.Substring(b.IndexOf('[')); b=b.Substring(0,b.LastIndexOf(']')+1);
        var culture=System.Globalization.CultureInfo.InvariantCulture;
        return "{\"profiles\":["+a+","+b+"],\"targetRadius\":"+radius.ToString(culture)+
            ",\"targetHp\":"+targetHp.ToString(culture)+",\"minDamage\":"+minDamage.ToString(culture)+
            ",\"maxDamage\":"+maxDamage.ToString(culture)+",\"criticalRadius\":"+criticalRadius.ToString(culture)+
            ",\"targetSpeed\":"+targetSpeed.ToString(culture)+",\"randomTargetSpeed\":"+(randomTargetSpeed?"true":"false")+
            ",\"targetSpeedMin\":"+targetSpeedMin.ToString(culture)+",\"targetSpeedMax\":"+targetSpeedMax.ToString(culture)+
            ",\"randomTargetSpawn\":"+(randomTargetSpawn?"true":"false")+
            ",\"shotIntervalsMs\":["+shotIntervals[0].ToString(culture)+","+shotIntervals[1].ToString(culture)+"],\"shotSpreadDeg\":["+
            shotSpreads[0].ToString(culture)+","+shotSpreads[1].ToString(culture)+"]}";
    }
    IEnumerator Run() {
        running=true; paused=false;
        while(running) {
            while(scoreResetBusy)yield return null;
            showingResult=false;status="Resetting BOTH agents"; round=null; clock=0;
            appliedEvents=0; exploded=false;
            if(debris!=null)Destroy(debris);
            target.gameObject.SetActive(true);
            foreach(var v in views){v.counted=false;v.criticalKill=false;v.shotStatus="";v.lastDamage=0;v.targetDamage=0;v.targetShots=0;v.lastShotTime=-999;v.bulletRay.enabled=false;}
            yield return Request("/round/reset",ResetJson(),p=>round=p);
            if(!running || round==null)break;
            SetRadius(round.agents[0].radius);
            yield return Advance();
            while(running && !round.done) {
                while((paused || scoreResetBusy) && running)yield return null;
                if(!running)break;
                status="LOCKSTEP: shared clock / inference barrier";
                yield return Request("/round/step","{}",p=>round=p);
                if(!running)break;
                yield return Advance();
            }
            if(!running)break;
            showingResult=true;
            status="타겟 파괴 · 결과 확인";
            RecordTarget();
            float remaining=resultPause;
            while(running && (remaining>0 || paused)) {
                if(!paused)remaining-=Time.unscaledDeltaTime;
                yield return null;
            }
            showingResult=false;
        }
        running=false;
    }
    void RecordTarget() {
        try {
            bool header=!File.Exists(resultLogPath);
            string row=DateTime.UtcNow.ToString("o")+","+round.round+","+clock.ToString("F1",CultureInfo.InvariantCulture);
            foreach(var v in views) row+=","+v.targetShots+","+v.targetDamage.ToString("G9",CultureInfo.InvariantCulture)+","+v.score.ToString("G9",CultureInfo.InvariantCulture);
            File.AppendAllText(resultLogPath,(header?"utc,target,time_ms,a_shots,a_damage,a_total_score,b_shots,b_damage,b_total_score\n":"")+row+"\n");
        } catch(Exception e) { Debug.LogError("Target result log: "+e.Message); }
    }
    IEnumerator ResetScores() {
        scoreResetBusy=true;
        while(modelRequestBusy)yield return null;
        float cutoff=clock;int currentRound=round==null?0:round.round;
        using(var req=new UnityWebRequest(Host+"/scores/reset","POST")) {
            req.uploadHandler=new UploadHandlerRaw(Encoding.UTF8.GetBytes(JsonUtility.ToJson(new ScoreResetRequest {round=currentRound,timeMs=cutoff})));
            req.downloadHandler=new DownloadHandlerBuffer();req.SetRequestHeader("Content-Type","application/json");req.timeout=15;
            yield return req.SendWebRequest();
            if(req.result==UnityWebRequest.Result.Success) {
                foreach(var v in views)v.score=0;
                if(round!=null) {
                    round.initialScores=new float[]{0,0};
                    foreach(var e in round.events)if(e.timeMs<=cutoff)e.scoreDelta=0;
                    ApplyAt(clock);
                }
            } else status="Score reset failed: "+req.error;
        }
        scoreResetBusy=false;
    }
    string AgentStatus(AgentView v) {
        if(v.packet==null)return "READY";
        if(visibleHp<=0)return "TARGET DESTROYED";
        return v.frame.cooldown?"COOLDOWN / TRACKING":"AIMING";
    }
    void DrawScoreboard() {
        Rect box=ScorePanel;
        GUI.Box(box,GUIContent.none,panelStyle);
        GUI.Label(new Rect(box.x+10,box.y+8,box.width-125,24),"경쟁 결과",headingStyle);
        GUI.enabled=!scoreResetBusy;
        if(GUI.Button(new Rect(box.xMax-110,box.y+8,100,24),scoreResetBusy?"초기화 중...":"점수 초기화"))StartCoroutine(ResetScores());
        GUI.enabled=true;
        var score=new GUIStyle(GUI.skin.label){fontSize=24,fontStyle=FontStyle.Bold};
        var detail=new GUIStyle(GUI.skin.label){fontSize=13,wordWrap=false};
        float y=box.y+38;
        for(int i=0;i<2;i++) {
            var v=views[i];GUI.contentColor=colors[i];
            GUI.Label(new Rect(box.x+10,y,box.width-20,34),"AGENT "+(i==0?"A":"B")+"   "+v.score.ToString("0.#"),score);
            GUI.contentColor=Color.white;
            GUI.Label(new Rect(box.x+10,y+34,box.width-20,22),"현재 타겟 발사 수  "+v.targetShots,detail);
            GUI.Label(new Rect(box.x+10,y+56,box.width-20,22),"현재 타겟 누적 피해  "+v.targetDamage.ToString("0.#"),detail);
            y+=82;
        }
        if(showingResult)GUI.Label(new Rect(box.x+10,y+2,box.width-20,24),"타겟 파괴 · 결과 기록 완료",detail);
    }

    class NumericEdit { public string text; public bool editing; }
    readonly Dictionary<string,NumericEdit> numericEdits=new Dictionary<string,NumericEdit>();
    float NumericSlider(string id,float value,float minimum,float maximum,bool integer=false,bool unbounded=false,float step=0) {
        if(!numericEdits.TryGetValue(id,out var edit)) {
            edit=new NumericEdit {text=value.ToString("G7",CultureInfo.InvariantCulture)};
            numericEdits[id]=edit;
        }
        if(step>0)value=Mathf.Round(value/step)*step;
        string control="numeric_"+id;
        bool focused=GUI.GetNameOfFocusedControl()==control;
        bool enter=focused && Event.current.type==EventType.KeyDown &&
            (Event.current.keyCode==KeyCode.Return || Event.current.keyCode==KeyCode.KeypadEnter);
        if((edit.editing && !focused) || enter) {
            if(float.TryParse(edit.text,NumberStyles.Float,CultureInfo.InvariantCulture,out float parsed) &&
                !float.IsNaN(parsed) && !float.IsInfinity(parsed))
                value=unbounded?Mathf.Max(minimum,integer?Mathf.Round(parsed):parsed):Mathf.Clamp(integer?Mathf.Round(parsed):parsed,minimum,maximum);
            if(step>0)value=Mathf.Clamp(Mathf.Round(value/step)*step,minimum,maximum);
            edit.text=value.ToString("G7",CultureInfo.InvariantCulture);
            if(enter){GUI.FocusControl(null);Event.current.Use();focused=false;}
        }
        if(!focused)edit.text=value.ToString("G7",CultureInfo.InvariantCulture);
        if(unbounded)maximum=Mathf.Max(maximum,value);
        GUILayout.BeginHorizontal();
        float slid=unbounded?value:GUILayout.HorizontalSlider(value,minimum,maximum,GUILayout.ExpandWidth(true));
        if(Mathf.Abs(slid-value)>.000001f) {
            value=step>0?Mathf.Clamp(Mathf.Round(slid/step)*step,minimum,maximum):(integer?Mathf.Round(slid):slid);
            edit.text=value.ToString("G7",CultureInfo.InvariantCulture);
        }
        GUI.SetNextControlName(control);
        string tooltip="Range: "+minimum.ToString("G7",CultureInfo.InvariantCulture)+" - "+maximum.ToString("G7",CultureInfo.InvariantCulture)+". Enter or leave field to apply.";
        // The visible interval also makes clamping predictable when entering a value.
        edit.text=unbounded?GUILayout.TextField(edit.text,24,GUILayout.ExpandWidth(true)):GUILayout.TextField(edit.text,24,GUILayout.Width(65));
        GUILayout.EndHorizontal();
        edit.editing=GUI.GetNameOfFocusedControl()==control;
        return unbounded?Mathf.Max(minimum,value):Mathf.Clamp(value,minimum,maximum);
    }
    void DrawTargetSettings() {
        BeginPanel(TargetPanel,"타겟 설정 · 다음 타겟부터 적용",ref targetScroll);
        randomTargetSpawn=GUILayout.Toggle(randomTargetSpawn,"Random starting position each target");
        GUILayout.Label("Radius (0.3-20 deg)");
        radius=NumericSlider("target_radius",radius,.3f,20);
        GUILayout.Label("Health (10-1000)");
        targetHp=NumericSlider("target_hp",targetHp,10,1000,true);
        GUILayout.Label("Instant-kill radius (0-30%)");
        criticalRadius=NumericSlider("target_critical",criticalRadius*100,0,30)/100;
        GUILayout.Label("Edge damage");
        minDamage=NumericSlider("target_edge",minDamage,0,100,false,true);
        maxDamage=Mathf.Max(maxDamage,minDamage);
        GUILayout.Label("Center damage");
        maxDamage=NumericSlider("target_center",maxDamage,minDamage,200,false,true);
        randomTargetSpeed=GUILayout.Toggle(randomTargetSpeed,"Random speed");
        if(randomTargetSpeed) {
            GUILayout.Label("Min speed (deg/s)");
            targetSpeedMin=NumericSlider("target_speed_min",targetSpeedMin,0,40);
            targetSpeedMax=Mathf.Max(targetSpeedMin,targetSpeedMax);
            GUILayout.Label("Max speed (deg/s)");
            targetSpeedMax=NumericSlider("target_speed_max",targetSpeedMax,targetSpeedMin,40);
        } else {
            GUILayout.Label("Motion speed (deg/s)");
            targetSpeed=NumericSlider("target_speed",targetSpeed,0,40);
        }

        if(round!=null)GUILayout.Label("Current speed "+round.motionSpeed.ToString("F2")+" deg/s");
        EndPanel();
    }
    void DrawAgentHUD(int i,Rect r) {
        var v=views[i];
        float width=Mathf.Min(420,r.width-24);
        float x=i==0?r.xMax-width-12:r.x+12;
        var alignment=i==0?TextAnchor.MiddleRight:TextAnchor.MiddleLeft;
        var small=new GUIStyle(GUI.skin.label){fontSize=12,wordWrap=false,alignment=alignment};
        GUI.color=colors[i];
        GUI.Label(new Rect(x,r.y+8,width,20),"AGENT "+(i==0?"A":"B")+" | "+AgentStatus(v),small);
        GUI.color=Color.white;
        GUI.Label(new Rect(x,r.y+28,width,20),v.shotStatus+" | 직전 피해 "+v.lastDamage.ToString("0.#"),small);
        if(v.packet!=null && v.packet.action!=null) {
            var a=v.packet.action;
            GUI.Label(new Rect(x,r.y+48,width,20),"Plan "+a.th.ToString("F0")+"ms | "+(v.frame.fireCommitted?"발사 결정":"발사하지 않음"),small);
            GUI.Label(new Rect(x,r.y+68,width,20),"Timing "+a.tc.ToString("F2")+" | Gaze "+a.kg.ToString("F2"),small);
        }
        var score=new GUIStyle(GUI.skin.label){fontSize=36,fontStyle=FontStyle.Bold,alignment=alignment,wordWrap=false};
        var count=new GUIStyle(score){fontSize=25};
        var damage=new GUIStyle(score){fontSize=20,fontStyle=FontStyle.Normal};
        GUI.color=colors[i];
        GUI.Label(new Rect(x,r.y+90,width,48),"SCORE  "+v.score.ToString("0.#"),score);
        GUI.color=Color.white;
        GUI.Label(new Rect(x,r.y+139,width,36),"발사 수  "+v.targetShots,count);
        GUI.Label(new Rect(x,r.y+175,width,30),"누적 피해  "+v.targetDamage.ToString("0.#"),damage);
        if(v.criticalKill && visibleHp<=0) {
            GUI.color=new Color(1,.8f,.15f);
            GUI.Label(new Rect(x,r.y+210,width,42),"CRITICAL",new GUIStyle(score){fontSize=30});
        }
        GUI.color=Color.white;
    }
    void OnGUI() {
        if(observer==null)return;
        if(uiFont!=null)GUI.skin.font=uiFont;
        if(panelStyle==null) {
            panelStyle=new GUIStyle(GUI.skin.box){padding=new RectOffset(10,10,8,8)};
            headingStyle=new GUIStyle(GUI.skin.label){fontStyle=FontStyle.Bold,wordWrap=true};
        }
        GUI.skin.label.wordWrap=true;
        if(Event.current.type==EventType.MouseDown && Event.current.button<=1 &&
            Event.current.mousePosition.y<Screen.height*.48f && !IsObserverUI(Event.current.mousePosition))
            GUI.FocusControl(null);
        DrawScoreboard();
        DrawTargetSettings();
        GUI.color=Color.white;
        BeginPanel(ControlPanel,"시뮬레이션",ref controlScroll);
        GUILayout.Label(status);
        GUILayout.Label("Round "+(round==null?0:round.round)+" | COMMON TIME "+clock.ToString("F0")+" ms");
        GUILayout.Label("TARGET HP "+visibleHp.ToString("F1")+" / "+(round==null?targetHp:round.maxHp).ToString("F0"));
        GUILayout.Label("SCORE   A "+views[0].score.ToString("0.#")+" : "+views[1].score.ToString("0.#")+" B");
        if(!running && GUILayout.Button("START BOTH AGENTS"))StartCoroutine(Run());
        if(running && GUILayout.Button(paused?"RESUME BOTH":"PAUSE BOTH"))paused=!paused;
        if(running && GUILayout.Button("STOP BOTH")){running=false;StopAllCoroutines();modelRequestBusy=false;scoreResetBusy=false;}
        GUILayout.Label("Playback speed (0.1-2x)"); speed=NumericSlider("playback",speed,.1f,2,step:.1f);
        showGaze=GUILayout.Toggle(showGaze,"Show each agent's gaze");
        GUILayout.Label("파괴 후 정지 (초)");resultPause=NumericSlider("result_pause",resultPause,0,5);
        GUILayout.Label("관찰 화면 클릭 후 WASD 이동 · 드래그 회전");
        EndPanel();
        for(int i=0;i<2;i++) {
            var v=views[i]; Rect r=personal[i];
            DrawAgentHUD(i,r);
            GUI.Label(new Rect(r.center.x-6,r.center.y-12,20,25),"+");
            if(showGaze) {GUI.color=new Color(1,.8f,.2f);GUI.Label(new Rect(r.x+r.width*(.5f+v.frame.gazeX/531.3f)-6,r.y+r.height*(.5f-v.frame.gazeY/298.8f)-12,50,24),"O");}
            GUI.color=Color.white;
            if(v.bulletRay.enabled) {
                Vector3 point=v.camera.WorldToScreenPoint(origins[i]+Direction(v.shotAz,v.shotEl)*20);
                Vector2 guiPoint=new Vector2(point.x,Screen.height-point.y);
                if(point.z>0 && r.Contains(guiPoint)) {
                    GUI.color=Color.yellow;GUI.Label(new Rect(guiPoint.x-6,guiPoint.y-12,20,24),"x");GUI.color=Color.white;
                }
            }
            if(target.gameObject.activeSelf) {
                Vector3 screen=v.camera.WorldToScreenPoint(target.position);
                if(screen.z>0)GUI.Label(new Rect(screen.x-48,Screen.height-screen.y-35,130,24),"HP "+visibleHp.ToString("F0"));
            }
            BeginPanel(AgentPanel(i),"AGENT "+(i==0?"A":"B")+" · 다음 타겟 설정",ref v.scroll);
            for(int j=0;j<8;j++) {
                if(j==4) {
                    GUILayout.Label("Shot cooldown (50-1000 ms)");
                    shotIntervals[i]=NumericSlider("agent_"+i+"_cooldown",shotIntervals[i],50,1000,true);
                    GUILayout.Label("Bullet spread (deg)");
                    shotSpreads[i]=NumericSlider("agent_"+i+"_spread",shotSpreads[i],0,180);
                    GUILayout.Space(6);
                }
                GUILayout.Label(labels[j]+(j<4?" (0-0.5)":(j<6?" (1-64)":" (5-95)")));
                profiles[i][j]=NumericSlider("agent_"+i+"_param_"+j,profiles[i][j],j<4?0:(j<6?1:5),j<4?.5f:(j<6?64:95));
            }
            GUILayout.Space(6);
            EndPanel();
        }
        numericFocused=(GUI.GetNameOfFocusedControl()??"").StartsWith("numeric_");
    }
}


