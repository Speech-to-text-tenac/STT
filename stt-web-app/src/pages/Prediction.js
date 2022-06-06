import classesp from "./Prediction.module.css";
import axios from "axios";
import { Link } from "react-router-dom";
import React, { useState } from "react";

let gumStream = null;
let recorder = null;
let audioContext = null;

export const Prediction = () => {
  const [show, setShow] = useState(false);
  const [view, setView] = useState(true);
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
      <div className={classesp.container}>
        <div className={classesp.link}>
          <span title="choose" onClick={() => setShow(!show)}></span>
          {/* <br></br>
        <span></span> */}
          {show ? (
            <ul>
              <li>
                <Link to="" onClick={() => setView(true)}>
                  Record Audio
                </Link>
              </li>
              <hr></hr>
              <li>
                <Link to="" onClick={() => setView(false)}>
                  Upload Audio
                </Link>
              </li>
            </ul>
          ) : null}
        </div>
        <div>
          {view ? (
            <div className={classesp.display}>
              <div>
                <main>
                <h2>Record Audio</h2>                
                <button onClick={startRecording} type="button">
                  Start
                </button>
                <button onClick={stopRecording} type="button">
                  Stop
                </button>
                </main>
                <br></br>
                <br></br>
                <audio autoplay controls></audio>
              </div>
              <section>
                <p>HELLO, EVERYONE</p>
              </section>
            </div>
          ) : (
            <div className={classesp.upload}>
              <div>
                <h2>Upload Audio</h2>
                <div>
                  <input type="file" />
                  <button>Predict</button>
                </div>
              </div>
              <section>
                <p>HELLO, EVERYONE</p>
              </section>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

