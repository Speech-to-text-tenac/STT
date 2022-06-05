import classesp from "./Prediction.module.css";
import axios from "axios";

let gumStream = null;
let recorder = null;
let audioContext = null;

export const Prediction = () => {
  // return (
  //   <>
  //     {/* <div> */}
  //     <div>
  //       <h2>Prediction</h2>
  //       <input type="file"></input>
  //       <button>predict</button>
  //     </div>

  //     <div>
  //       <h2>output</h2>
  //     </div>
  //     {/* </div> */}
  //   </>
  // );
  const startRecording = () => {
    let constraints = {
      audio: true,
      video: false,
    };
  
    audioContext = new window.AudioContext();
    console.log("sample rate: " + audioContext.sampleRate);

    navigator.mediaDevices
      .getUserMedia(constraints)
      .then(function (stream) {
        console.log("initializing Recorder.js ...");

        gumStream = stream;

        let input = audioContext.createMediaStreamSource(stream);
        console.log(input)
        
        recorder = new window.Recorder(input, {
          numChannels: 1,
        });
        
        console.log(recorder);
        recorder.record();
        console.log("Recording started");
      })
      .catch(function (err) {
        //enable the record button if getUserMedia() fails
        console.log(err)
      });
  };

  const stopRecording = () => {
    console.log("stopButton clicked");

    recorder.stop(); //stop microphone access
    gumStream.getAudioTracks()[0].stop();

    recorder.exportWAV(onStop);
  };

  const onStop = (blob) => {
    console.log("uploading...");

    let data = new FormData();

    data.append("text", "this is the transcription of the audio file");
    data.append("wavfile", blob, "recording.wav");

    const config = {
      headers: { "content-type": "multipart/form-data" },
    };
    console.log(data)
    axios.post("http://localhost:8080/asr/", data, config);
  };

  return (
    <>
      {/* <div className={classesp.display}>
        <button onClick={startRecording} type="button">
          Start
        </button>
        <button onClick={stopRecording} type="button">
          Stop
        </button>
        <br></br>
        <br></br>
        <audio autoplay controls></audio>

        <section>
          <p>HELLO, EVERYONE</p>
        </section>
      </div> */}

      <div className={classesp.upload}>
        <h2>Upload Audio</h2>
        <section>
          <input type="file" />
          <button>Predict</button>
        </section>
      </div>
    </>
  );
};

