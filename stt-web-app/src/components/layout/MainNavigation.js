import { Link } from "react-router-dom";
import classes from "./MainNavigation.module.css";
// classes in the above can be any name we want

export const MainNavigation = () => {
  return (
    <>
      <div className={classes.nav_contain}>
        <header className={classes.header}>
          <div className={classes.logo}>
            <h4>Speech Recongintion</h4>
          </div>
          <nav className={classes.container}>
          <ul>
            <li>
              <Link to="/">Home</Link>
            </li>
            <li>
              <Link to="/prediction">Prediction</Link>
            </li>
            {/* <li>
              <Link to="/visual">Visualization</Link>
            </li> */}
            <li>
              <Link to="/about-us">About Us</Link>
            </li>
          </ul>
        </nav>
        </header>

       
      </div>
    </>
  );
};
