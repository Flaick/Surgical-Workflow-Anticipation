import numpy as np
import os
from options import parser
import util as util



def test_scores(opts, train=True):
	horizon = opts.horizon

	# load predictions (NUM_INSTRUMENTS x NUM_FRAMES x NUM_SAMPLES) and targets (NUM_INSTRUMENTS x NUM_FRAMES)
	prediction_tool, target_tool, prediction_phase, target_phase = util.load_samples_ins_mstcn(opts)

	# use sample mean as an estimate for the predictive expectation
	# now prediction and target both have shape (NUM_INSTRUMENTS x NUM_FRAMES)

	########################### Metrics for Tool ###############################
	print('########################### Metrics for Tool ###############################')
	wMAE = []
	out_MAE = []
	in_MAE = []
	pMAE = []
	eMAE = []
	dMAE = []
	mMAE = []

	for y, t in zip(prediction_tool, target_tool):
		outside_horizon = (t == horizon)
		inside_horizon = (t < horizon) & (t > 0)
		anticipating = (y > horizon*.1) & (y < horizon*.9)

		distant_anticipating = (t > horizon*.9) & (t < horizon)
		e_anticipating = (t < horizon*.1) & (t > 0)
		m_anticipating = (t > horizon*.1) & (t < horizon*.9)


		wMAE_ins = np.mean([
			np.abs(y[outside_horizon]-t[outside_horizon]).mean(),
			np.abs(y[inside_horizon]-t[inside_horizon]).mean()
		])
		out_MAE_ins = np.mean([np.abs(y[outside_horizon]-t[outside_horizon]).mean()])
		in_MAE_ins = np.mean([np.abs(y[inside_horizon]-t[inside_horizon]).mean()])
		pMAE_ins = np.abs(y[anticipating]-t[anticipating]).mean()
		eMAE_ins = np.abs(y[e_anticipating]-t[e_anticipating]).mean()
		dMAE_ins = np.abs(y[distant_anticipating]-t[distant_anticipating]).mean()
		mMAE_ins = np.abs(y[m_anticipating]-t[m_anticipating]).mean()


		wMAE.append(wMAE_ins)
		out_MAE.append(out_MAE_ins)
		in_MAE.append(in_MAE_ins)
		pMAE.append(pMAE_ins)
		eMAE.append(eMAE_ins)
		dMAE.append(dMAE_ins)
		mMAE.append(mMAE_ins)

	if train:
		return np.mean(wMAE), np.mean(out_MAE), np.mean(in_MAE), np.mean(pMAE), np.mean(eMAE), np.mean(dMAE), np.mean(mMAE)
	else:
		util.print_scores(instruments=util.instruments, scores=wMAE, header='== wMAE [min.] ==')
		util.print_scores(instruments=util.instruments, scores=out_MAE, header='== out MAE [min.] ==')
		util.print_scores(instruments=util.instruments, scores=in_MAE, header='== in MAE [min.] ==')
		util.print_scores(instruments=util.instruments, scores=pMAE, header='== pMAE [min.] ==')
		util.print_scores(instruments=util.instruments, scores=eMAE, header='== eMAE [min.] ==')
		util.print_scores(instruments=util.instruments, scores=dMAE, header='== dMAE [min.] ==')
		util.print_scores(instruments=util.instruments, scores=mMAE, header='== mMAE [min.] ==')

	########################### Metrics for Phase ###############################\
	print('########################### Metrics for Phase ###############################')
	wMAE = []
	out_MAE = []
	in_MAE = []
	pMAE = []
	eMAE = []
	dMAE = []
	mMAE = []

	for y, t in zip(prediction_phase, target_phase):

		outside_horizon = (t == horizon)
		inside_horizon = (t < horizon) & (t > 0)
		anticipating = (y > horizon*.1) & (y < horizon*.9)

		distant_anticipating = (t > horizon*.9) & (t < horizon)
		e_anticipating = (t < horizon*.1) & (t > 0)
		m_anticipating = (t > horizon*.1) & (t < horizon*.9)

		wMAE_ins = np.mean([
			np.abs(y[outside_horizon]-t[outside_horizon]).mean(),
			np.abs(y[inside_horizon]-t[inside_horizon]).mean()
		])
		out_MAE_ins = np.mean([np.abs(y[outside_horizon]-t[outside_horizon]).mean()])
		in_MAE_ins = np.mean([np.abs(y[inside_horizon]-t[inside_horizon]).mean()])
		pMAE_ins = np.abs(y[anticipating]-t[anticipating]).mean()
		eMAE_ins = np.abs(y[e_anticipating]-t[e_anticipating]).mean()
		dMAE_ins = np.abs(y[distant_anticipating]-t[distant_anticipating]).mean()
		mMAE_ins = np.abs(y[m_anticipating]-t[m_anticipating]).mean()


		wMAE.append(wMAE_ins)
		out_MAE.append(out_MAE_ins)
		in_MAE.append(in_MAE_ins)
		pMAE.append(pMAE_ins)
		eMAE.append(eMAE_ins)
		dMAE.append(dMAE_ins)
		mMAE.append(mMAE_ins)

	if train:
		return np.mean(wMAE), np.mean(out_MAE), np.mean(in_MAE), np.mean(pMAE), np.mean(eMAE), np.mean(dMAE), np.mean(mMAE)
	else:
		util.print_scores(instruments=util.phases, scores=wMAE, header='== wMAE [min.] ==')
		util.print_scores(instruments=util.phases, scores=out_MAE, header='== out MAE [min.] ==')
		util.print_scores(instruments=util.phases, scores=in_MAE, header='== in MAE [min.] ==')
		util.print_scores(instruments=util.phases, scores=pMAE, header='== pMAE [min.] ==')
		util.print_scores(instruments=util.phases, scores=eMAE, header='== eMAE [min.] ==')
		util.print_scores(instruments=util.phases, scores=dMAE, header='== dMAE [min.] ==')
		util.print_scores(instruments=util.phases, scores=mMAE, header='== mMAE [min.] ==')

if __name__ == '__main__':
	opts = parser.parse_args()
	test_scores(opts, train=False)



