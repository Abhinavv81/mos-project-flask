from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        n_forces = int(request.form["n_forces"])
        forces = []
        for i in range(n_forces):
            position = float(request.form[f"position_{i}"])
            magnitude = float(request.form[f"magnitude_{i}"])
            forces.append((position, magnitude))

        L = float(request.form["beam_length"])
        w = float(request.form["udl_magnitude"])
        start_w = float(request.form["udl_start"])
        end_w = float(request.form["udl_end"])

        total_load = sum([magnitude for _, magnitude in forces])
        udl_load = w * (end_w - start_w)
        total_vertical_load = total_load + udl_load

        moment_A = sum([magnitude * position for position, magnitude in forces]) + (
            (w * (end_w - start_w)) * ((end_w + start_w) / 2)
        )

        RB = moment_A / L
        RA = total_vertical_load - RB

        def shear_force(x):
            sf = RA
            if x >= start_w:
                sf -= w * min(x - start_w, end_w - start_w)
            for pos, force in forces:
                if x >= pos:
                    sf -= force
            return sf

        def bending_moment(x):
            bm = RA * x
            for pos, force in forces:
                if x >= pos:
                    bm -= force * (x - pos)
            if start_w <= x <= end_w:
                bm -= (w * (x - start_w)) * ((x - start_w) / 2)
            return bm

        x_vals = np.sort(
            np.concatenate(
                [
                    np.linspace(0, L, 100),
                    np.array([pos - 0.001 for pos, _ in forces if pos > 0]),
                    np.array([pos + 0.001 for pos, _ in forces if pos < L]),
                ]
            )
        )
        sf_vals = [shear_force(x) for x in x_vals]
        bm_vals = [bending_moment(x) for x in x_vals]

        # Generate plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        ax1.plot(x_vals, sf_vals, label="Shear Force", color="blue")
        ax1.fill_between(
            x_vals,
            0,
            sf_vals,
            where=(np.array(sf_vals) > 0),
            color="blue",
            alpha=0.3,
            interpolate=True,
        )
        ax1.fill_between(
            x_vals,
            0,
            sf_vals,
            where=(np.array(sf_vals) < 0),
            color="red",
            alpha=0.3,
            interpolate=True,
        )
        ax1.set_title("Shear Force Diagram")
        ax1.set_xlabel("Position along the beam (m)")
        ax1.set_ylabel("Shear Force (kN)")
        ax1.grid(True)

        ax2.plot(x_vals, bm_vals, label="Bending Moment", color="green")
        ax2.fill_between(
            x_vals,
            0,
            bm_vals,
            where=(np.array(bm_vals) >= 0),
            color="green",
            alpha=0.3,
            interpolate=True,
        )
        ax2.fill_between(
            x_vals,
            0,
            bm_vals,
            where=(np.array(bm_vals) <= 0),
            color="orange",
            alpha=0.3,
            interpolate=True,
        )
        ax2.set_title("Bending Moment Diagram")
        ax2.set_xlabel("Position along the beam (m)")
        ax2.set_ylabel("Bending Moment (kNm)")
        ax2.grid(True)

        plt.tight_layout()

        # Save plots to a BytesIO object and send as response
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plt.close()

        return send_file(img, mimetype="image/png")

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
