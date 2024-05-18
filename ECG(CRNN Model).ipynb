{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8667c8b5-8b41-4346-99dc-09d51d75cb10",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import wfdb \n",
    "from collections import Counter\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from imblearn.over_sampling import SMOTENC\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "#from ecgdetectors import Detectors\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4922db7f-d5dd-44ad-b097-9bca80738a07",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   ecg_id  patient_id   age  sex  height  weight  nurse  site      device  \\\n",
      "0    4759      3250.0  39.0    1   180.0   116.0    NaN  41.0  AT-6 C 5.5   \n",
      "1    9278     19350.0  60.0    1     NaN     NaN    0.0   0.0  CS100    3   \n",
      "2   13398     20811.0  65.0    1     NaN     NaN    1.0   2.0       CS-12   \n",
      "3   11144      1160.0  55.0    0   183.0    77.0    5.0   1.0  AT-6     6   \n",
      "4     399     15234.0  40.0    0     NaN    85.0    2.0   0.0   CS-12   E   \n",
      "\n",
      "        recording_date  ... validated_by_human baseline_drift static_noise  \\\n",
      "0  1990-08-11 14:18:54  ...               True            NaN          NaN   \n",
      "1  1992-11-03 12:04:22  ...              False            NaN          NaN   \n",
      "2  1994-11-29 11:34:41  ...               True            NaN          NaN   \n",
      "3  1993-10-02 07:45:26  ...               True           , v1          NaN   \n",
      "4  1987-01-17 14:16:48  ...               True            NaN          NaN   \n",
      "\n",
      "  burst_noise electrodes_problems  extra_beats  pacemaker  strat_fold  \\\n",
      "0         NaN                 NaN          NaN        NaN           5   \n",
      "1         NaN                 NaN          NaN        NaN           7   \n",
      "2         NaN                 NaN          NaN        NaN          10   \n",
      "3         NaN                 NaN          NaN        NaN          10   \n",
      "4         NaN                 NaN          NaN        NaN           4   \n",
      "\n",
      "                 filename_lr                filename_hr  \n",
      "0  records100/04000/04759_lr  records500/04000/04759_hr  \n",
      "1  records100/09000/09278_lr  records500/09000/09278_hr  \n",
      "2  records100/13000/13398_lr  records500/13000/13398_hr  \n",
      "3  records100/11000/11144_lr  records500/11000/11144_hr  \n",
      "4  records100/00000/00399_lr  records500/00000/00399_hr  \n",
      "\n",
      "[5 rows x 28 columns]\n"
     ]
    }
   ],
   "source": [
    "datasetdf_path = 'New data.csv'\n",
    "# Read dataset into a DataFrame\n",
    "datasetdf_df = pd.read_csv(datasetdf_path)\n",
    "\n",
    "# Display the first few rows of the DataFrame\n",
    "print(datasetdf_df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a1e379b-4f16-4c55-80e5-f1f7400bf7a6",
   "metadata": {},
   "source": [
    "#Code"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06873073-bab6-4e44-8c2a-20b89c84cc17",
   "metadata": {},
   "source": [
    "###Code"
   ]
  },
  {
   "cell_type": "raw",
   "id": "d6e8eb51-e93f-4280-a1e4-c4f80ea20a6d",
   "metadata": {},
   "source": [
    "Code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fd2edab7-48f5-4fa0-b9a7-38ac51efe677",
   "metadata": {},
   "outputs": [],
   "source": [
    "def collect_and_label(dataset):\n",
    "    # Collect only the MI classes and the NORM classes from the dataset.\n",
    "    df = pd.read_csv(dataset)\n",
    "    alpha = df['scp_codes'].str.split(\"'\").str[1].str[-2:] == 'MI'  # Collect all the MI classes.\n",
    "    beta = df['scp_codes'].str.split(\"'\").str[1] == 'NORM'  # Collect all the Normal classes.\n",
    "    df = df[alpha | beta]\n",
    "    df['label'] = df['scp_codes'].str.split(\"'\").str[1]  # Create a new column 'label' containing categorical labels.\n",
    "\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d2f2b2f5-4f5e-463e-a9b8-6c802f4aa203",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   ecg_id  patient_id   age  sex  height  weight  nurse  site      device  \\\n",
      "0    4759      3250.0  39.0    1   180.0   116.0    NaN  41.0  AT-6 C 5.5   \n",
      "1    9278     19350.0  60.0    1     NaN     NaN    0.0   0.0  CS100    3   \n",
      "2   13398     20811.0  65.0    1     NaN     NaN    1.0   2.0       CS-12   \n",
      "3   11144      1160.0  55.0    0   183.0    77.0    5.0   1.0  AT-6     6   \n",
      "4     399     15234.0  40.0    0     NaN    85.0    2.0   0.0   CS-12   E   \n",
      "\n",
      "        recording_date  ... baseline_drift static_noise burst_noise  \\\n",
      "0  1990-08-11 14:18:54  ...            NaN          NaN         NaN   \n",
      "1  1992-11-03 12:04:22  ...            NaN          NaN         NaN   \n",
      "2  1994-11-29 11:34:41  ...            NaN          NaN         NaN   \n",
      "3  1993-10-02 07:45:26  ...           , v1          NaN         NaN   \n",
      "4  1987-01-17 14:16:48  ...            NaN          NaN         NaN   \n",
      "\n",
      "  electrodes_problems extra_beats  pacemaker  strat_fold  \\\n",
      "0                 NaN         NaN        NaN           5   \n",
      "1                 NaN         NaN        NaN           7   \n",
      "2                 NaN         NaN        NaN          10   \n",
      "3                 NaN         NaN        NaN          10   \n",
      "4                 NaN         NaN        NaN           4   \n",
      "\n",
      "                 filename_lr                filename_hr label  \n",
      "0  records100/04000/04759_lr  records500/04000/04759_hr  NORM  \n",
      "1  records100/09000/09278_lr  records500/09000/09278_hr  NORM  \n",
      "2  records100/13000/13398_lr  records500/13000/13398_hr  NORM  \n",
      "3  records100/11000/11144_lr  records500/11000/11144_hr  NORM  \n",
      "4  records100/00000/00399_lr  records500/00000/00399_hr  NORM  \n",
      "\n",
      "[5 rows x 29 columns]\n"
     ]
    }
   ],
   "source": [
    "df_labeled = collect_and_label('New data.csv')\n",
    "print(df_labeled.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c339ba2d-6b91-41e3-b2e8-6a8f7a34f7a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Label 'NORM' has 2232 records.\n",
      "Label 'IMI' has 1943 records.\n",
      "Label 'ASMI' has 232 records.\n",
      "Label 'PMI' has 1 records.\n",
      "Label 'AMI' has 41 records.\n",
      "Label 'ALMI' has 8 records.\n",
      "Label 'LMI' has 1 records.\n"
     ]
    }
   ],
   "source": [
    "label_counts = Counter(df_labeled['label'])\n",
    "for label, count in label_counts.items():\n",
    "    print(f\"Label '{label}' has {count} records.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "db0a110e-982e-4dfb-b084-7afbf77e3f6e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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hxEiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeYUaiNatW6dOnTrJz89PNptNixcvtlvft29f2Ww2u0f79u3t2pw5c0Y9e/aUp6envL29FR4ertTUVLs2v/32m1q2bCkXFxf5+/tr3Lhx+d01AABQjBRqILpw4YLuv/9+TZ069ZZt2rdvr4SEBPPx9ddf263v2bOndu3apdjYWC1ZskTr1q3T888/b65PSUlRu3btFBAQoJ9//lnjx49XVFSUvvjii3zrFwAAKF4cC/PNg4ODFRwcfNs2zs7O8vHxyXbdnj17tHTpUv3000968MEHJUlTpkxRhw4d9MEHH8jPz09fffWVrl69qunTp8vJyUn16tXT9u3bNXHiRLvgBAAArKvIzyFas2aNKlSooFq1aumFF17Q6dOnzXVxcXHy9vY2w5AktW3bViVKlNDmzZvNNq1atZKTk5PZJigoSHv37tXZs2ezfc8rV64oJSXF7gEAAO5dRToQtW/fXjNnztTKlSs1duxYrV27VsHBwUpPT5ckJSYmqkKFCnavcXR0VJkyZZSYmGi2qVixol2bzOeZbW42ZswYeXl5mQ9/f/+87hoAAChCCvWU2Z2Ehoaa/2/QoIEaNmyowMBArVmzRm3atMm393399dcVGRlpPk9JSSEUAQBwDyvSI0Q3q1atmsqVK6cDBw5Iknx8fHTixAm7NteuXdOZM2fMeUc+Pj5KSkqya5P5/FZzk5ydneXp6Wn3AAAA965iFYiOHj2q06dPy9fXV5LUvHlznTt3Tj///LPZZtWqVcrIyFDTpk3NNuvWrVNaWprZJjY2VrVq1VLp0qULtgMAAKBIKtRAlJqaqu3bt2v79u2SpEOHDmn79u2Kj49Xamqqhg0bpk2bNunw4cNauXKlnnrqKVWvXl1BQUGSpDp16qh9+/YaOHCgtmzZog0bNmjQoEEKDQ2Vn5+fJKlHjx5ycnJSeHi4du3apTlz5uijjz6yOyUGAACsrVAD0datW9W4cWM1btxYkhQZGanGjRtrxIgRcnBw0G+//aYnn3xSNWvWVHh4uJo0aaIff/xRzs7O5ja++uor1a5dW23atFGHDh3UokULu3sMeXl5afny5Tp06JCaNGmiIUOGaMSIEVxyDwAATIU6qbp169YyDOOW65ctW3bHbZQpU0azZs26bZuGDRvqxx9/zHF9AADAGorVHCIAAID8QCACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWl6tA9Mcff+R1HQAAAIUmV4GoevXqeuyxx/Sf//xHly9fzuuaAAAAClSuAtG2bdvUsGFDRUZGysfHR3//+9+1ZcuWvK4NAACgQOQqEDVq1EgfffSRjh8/runTpyshIUEtWrRQ/fr1NXHiRJ08eTKv6wQAAMg3f2lStaOjo7p27ap58+Zp7NixOnDggIYOHSp/f3/16dNHCQkJeVUnAABAvvlLgWjr1q168cUX5evrq4kTJ2ro0KE6ePCgYmNjdfz4cT311FN5VScAAEC+cczNiyZOnKjo6Gjt3btXHTp00MyZM9WhQweVKHE9X1WtWlUxMTGqUqVKXtYKAACQL3IViD799FP1799fffv2la+vb7ZtKlSooGnTpv2l4gAAAApCrgLR/v3779jGyclJYWFhudk8AABAgcrVHKLo6GjNmzcvy/J58+ZpxowZf7koAACAgpSrQDRmzBiVK1cuy/IKFSrovffe+8tFAQAAFKRcBaL4+HhVrVo1y/KAgADFx8f/5aIAAAAKUq4CUYUKFfTbb79lWf7rr7+qbNmyf7koAACAgpSrQNS9e3f985//1OrVq5Wenq709HStWrVKL7/8skJDQ/O6RgAAgHyVq6vM3nnn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0dFRPXr00PTp081lR48e1Zo1a9SjR4+C6kq+cHZ2lo+PjwICAvTCCy+obdu2+vbbb831YWFhdv2+dOmSZs+erbCwsMIoN1/c7hjJFBYWluUHYHR0dLHfD3fT99u5lz8bt9OzZ0+tXbtWf/75p7ls+vTp6tmzpxwd781bGBKIijgHBwe99957mjJlio4ePZpl/axZs1SzZk116tQpy7ohQ4bo9OnTio2NzXbbJ06c0KJFi+Tg4CAHBwe7dU5OTurZs6fdN8iYmBj179//L/aocPTv39+uL9OnT79n/mzL3LlzVbt2bdWqVUu9evXS9OnTlXl7MR8fH+3atUs7d+6843b69++vuXPn6uLFi5Kuf73bt29/z93N3dXVVVevXjWf9+7dWz/++KN56njBggWqUqWKHnjggcIqMc/d7hjJ9OSTT+rs2bNav369JGn9+vU6e/Zstt9bipO76fudWOWzcaOKFSsqKChIM2bMkCRdvHhRc+bMKbY/A+4GgagY6NKlixo1aqSRI0dmWbdv375sz29LMpfv27fPXJacnCx3d3e5ubmpYsWKWr16tSIiIrIdAs38JnDhwgWtW7dOycnJ6tixYx71qmD16tVL69ev15EjR3TkyBFt2LBBvXr1Kuyy8sS0adPMvrRv317Jyclau3atJOmll17S3/72NzVo0EBVqlRRaGiopk+fritXrmTZTuPGjVWtWjXNnz9fhmEU6wCcHcMwtGLFCi1btkyPP/64ubxChQoKDg42Rw6mT59+T/Vbuv0xkqlkyZJmYJCu74devXqpZMmSBV5vXrqbvt/Jvf7ZuJX+/fsrJiZGhmFo/vz5CgwMVKNGjQq7rHxDIComxo4dqxkzZmjPnj1Z1uXktx0PDw9t375dW7du1YQJE/TAAw9o9OjR2ba9//77VaNGDc2fP1/Tp09X7969i+1Qafny5c3h8ujoaIWEhKhcuXKFXdZftnfvXm3ZskXdu3eXdH14/7nnntO0adMkXZ//8P333+vAgQN688035e7uriFDhuihhx4yf9u9UeZI2tq1a3XhwgV16NChQPuTH5YsWSJ3d3e5uLgoODhYzz33XJaLAjK/8f/xxx+Ki4tTz549C6fYfHCnY+RG/fv317x585SYmKh58+YV+x/6Oen7ndyLn407CQkJUWpqqtatW3dP/qJws+L5082CWrVqpaCgIL3++ut2V0XVrFkz25AkyVxes2ZNc1mJEiVUvXp1SddHkA4ePKgXXnhBX375Zbbb6N+/v6ZOnardu3dry5YtedSbwtG/f3/zb9ndPIemuJo2bZquXbsmPz8/c5lhGHJ2dtbHH38sLy8vSVJgYKACAwM1YMAAvfHGG6pZs6bmzJmT5bRhz5499eqrryoqKqpYB+AbPfbYY/r000/l5OQkPz+/bPsUHBys559/XuHh4erUqZPKli1bCJXmjzsdIzdq0KCBateure7du6tOnTqqX79+sbqa7GY56fud3IufjTtxdHRU7969NXLkSG3evDnLxQf3GkaIipH3339f3333nd3VDqGhodq/f7++++67LO0nTJigsmXL6oknnrjlNl977TXNmTNH27Zty3Z9jx49tGPHDtWvX19169b9650oRO3bt9fVq1eVlpamoKCgwi7nL7t27ZpmzpypCRMmaPv27ebj119/lZ+fn77++utsX1elShWVKlXKbrJ9pjJlyujJJ5/U2rVr75nfBt3c3FS9enVVrlz5lj/EHB0d1adPH61Zs+ae6beUu2Okf//+98R+yO3n41buxc/G3ejfv7/Wrl2rp556SqVLly7scvLVvR9x7yENGjRQz549NXnyZHNZaGio5s2bp7CwMI0fP15t2rRRSkqKpk6dqm+//Vbz5s277SWS/v7+6tKli0aMGKElS5ZkWV+6dGklJCQU+3kE0vUJ6pmjZjdPIi+OlixZorNnzyo8PNwcCcrUrVs3TZs2TYmJibp48aI6dOiggIAAnTt3TpMnT1ZaWtotg3JMTIw++eSTe2qU5G688847GjZs2D3V77s5Rtq3b2+3fODAgXrmmWeKzf10biUnfd+7d2+W19erVy/Lsnvts5GcnJxlBPDmvtWpU0enTp1SqVKlCrCywkEgKmZGjRqlOXPmmM9tNpvmzp2rDz/8UJMmTdKLL74oFxcXNW/eXGvWrNEjjzxyx22+8sorat68ubZs2aKHHnooy/ri/o3xRp6enoVdQp6ZNm2a2rZtm+WbvXT9G/64cePUq1cv7dy5U3369FFSUpJKly6txo0ba/ny5apVq1a223V1dZWrq2t+l1/kODk53RPzym50N8fIzbfccHR0vCf2Q076HhoamqXNjZebZ7rXPhtr1qzJcuuR8PDwLO3ulQB4JzYjp9cfAgAA3GOYQwQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQATgnmCz2bR48eLCLgNAMUUgAlAsJCYm6qWXXlK1atXk7Owsf39/derUSStXrizs0gDcA/jTHQCKvMOHD+uRRx6Rt7e3xo8frwYNGigtLU3Lli1TRESEfv/998IuEUAxxwgRgCLvxRdflM1m05YtW9StWzfVrFlT9erVU2RkpDZt2pTta4YPH66aNWuqVKlSqlatmt566y2lpaWZ63/99Vc99thj8vDwkKenp5o0aaKtW7dKko4cOaJOnTqpdOnScnNzU7169fTf//7XfO3OnTsVHBwsd3d3VaxYUb1799apU6fM9fPnz1eDBg3k6uqqsmXLqm3btrpw4UI+7R0AeYERIgBF2pkzZ7R06VKNHj1abm5uWdbf6o8Pe3h4KCYmRn5+ftqxY4cGDhwoDw8Pvfrqq5Kknj17qnHjxvr000/l4OCg7du3q2TJkpKkiIgIXb16VevWrZObm5t2794td3d3SdK5c+f0+OOPa8CAAZo0aZIuXbqk4cOH69lnn9WqVauUkJCg7t27a9y4cerSpYvOnz+vH3/8UfzZSKBoIxABKNIOHDggwzBUu3btHL3uzTffNP9fpUoVDR06VLNnzzYDUXx8vIYNG2Zut0aNGmb7+Ph4devWTQ0aNJAkVatWzVz38ccfq3HjxnrvvffMZdOnT5e/v7/27dun1NRUXbt2TV27dlVAQIAkmdsBUHQRiAAUabkdWZkzZ44mT56sgwcPmiHF09PTXB8ZGakBAwboyy+/VNu2bfXMM88oMDBQkvTPf/5TL7zwgpYvX662bduqW7duatiwoaTrp9pWr15tjhjd6ODBg2rXrp3atGmjBg0aKCgoSO3atdPTTz+t0qVL56ofAAoGc4gAFGk1atSQzWbL0cTpuLg49ezZUx06dNCSJUv0yy+/6I033tDVq1fNNlFRUdq1a5dCQkK0atUq1a1bV4sWLZIkDRgwQH/88Yd69+6tHTt26MEHH9SUKVMkSampqerUqZO2b99u99i/f79atWolBwcHxcbG6ocfflDdunU1ZcoU1apVS4cOHcrbHQMgT9kMTmwDKOKCg4O1Y8cO7d27N8s8onPnzsnb21s2m02LFi1S586dNWHCBH3yySc6ePCg2W7AgAGaP3++zp07l+17dO/eXRcuXNC3336bZd3rr7+u77//Xr/99pveeOMNLViwQDt37pSj450H2dPT0xUQEKDIyEhFRkbmrOMACgwjRACKvKlTpyo9PV0PPfSQFixYoP3792vPnj2aPHmymjdvnqV9jRo1FB8fr9mzZ+vgwYOaPHmyOfojSZcuXdKgQYO0Zs0aHTlyRBs2bNBPP/2kOnXqSJIGDx6sZcuW6dChQ9q2bZtWr15trouIiNCZM2fUvXt3/fTTTzp48KCWLVumfv36KT09XZs3b9Z7772nrVu3Kj4+XgsXLtTJkyfN1wMoogwAKAaOHz9uREREGAEBAYaTk5NRqVIl48knnzRWr15tGIZhSDIWLVpkth82bJhRtmxZw93d3XjuueeMSZMmGV5eXoZhGMaVK1eM0NBQw9/f33BycjL8/PyMQYMGGZcuXTIMwzAGDRpkBAYGGs7Ozkb58uWN3r17G6dOnTK3vW/fPqNLly6Gt7e34erqatSuXdsYPHiwkZGRYezevdsICgoyypcvbzg7Oxs1a9Y0pkyZUlC7CUAuccoMAABYHqfMAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5f0/zu/zJBOGnDoAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.bar(label_counts.keys(), label_counts.values())\n",
    "plt.xlabel('Classes')\n",
    "plt.ylabel('Frequency')\n",
    "plt.title('Frequency of Each Class in the Dataset')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "925eaee6-aa73-449f-97e1-7916e3c05f28",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   ecg_id  patient_id   age  sex  height  weight  nurse  site      device  \\\n",
      "0    4759      3250.0  39.0    1   180.0   116.0    NaN  41.0  AT-6 C 5.5   \n",
      "1    9278     19350.0  60.0    1     NaN     NaN    0.0   0.0  CS100    3   \n",
      "2   13398     20811.0  65.0    1     NaN     NaN    1.0   2.0       CS-12   \n",
      "3   11144      1160.0  55.0    0   183.0    77.0    5.0   1.0  AT-6     6   \n",
      "4     399     15234.0  40.0    0     NaN    85.0    2.0   0.0   CS-12   E   \n",
      "\n",
      "        recording_date  ... baseline_drift static_noise burst_noise  \\\n",
      "0  1990-08-11 14:18:54  ...            NaN          NaN         NaN   \n",
      "1  1992-11-03 12:04:22  ...            NaN          NaN         NaN   \n",
      "2  1994-11-29 11:34:41  ...            NaN          NaN         NaN   \n",
      "3  1993-10-02 07:45:26  ...           , v1          NaN         NaN   \n",
      "4  1987-01-17 14:16:48  ...            NaN          NaN         NaN   \n",
      "\n",
      "  electrodes_problems extra_beats  pacemaker  strat_fold  \\\n",
      "0                 NaN         NaN        NaN           5   \n",
      "1                 NaN         NaN        NaN           7   \n",
      "2                 NaN         NaN        NaN          10   \n",
      "3                 NaN         NaN        NaN          10   \n",
      "4                 NaN         NaN        NaN           4   \n",
      "\n",
      "                 filename_lr                filename_hr label  \n",
      "0  records100/04000/04759_lr  records500/04000/04759_hr  NORM  \n",
      "1  records100/09000/09278_lr  records500/09000/09278_hr  NORM  \n",
      "2  records100/13000/13398_lr  records500/13000/13398_hr  NORM  \n",
      "3  records100/11000/11144_lr  records500/11000/11144_hr  NORM  \n",
      "4  records100/00000/00399_lr  records500/00000/00399_hr  NORM  \n",
      "\n",
      "[5 rows x 29 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Assuming your dataset is stored in a DataFrame called 'df'\n",
    "# and the label column is named 'label'\n",
    "\n",
    "# Filter the data to include only records with labels 'NORM' and 'IMI'\n",
    "data = df_labeled[df_labeled['label'].isin(['NORM', 'IMI'])]\n",
    "\n",
    "# Display the first few rows of the filtered data to verify\n",
    "print(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2a687b11-cf39-4af4-bf54-28d41e6084ef",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Label 'NORM' has 2232 records.\n",
      "Label 'IMI' has 1943 records.\n"
     ]
    }
   ],
   "source": [
    "label_counts = Counter(data['label'])\n",
    "for label, count in label_counts.items():\n",
    "    print(f\"Label '{label}' has {count} records.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b55b7b94-6301-4ae9-bd6b-d5716af94373",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "from imblearn.over_sampling import SMOTENC\n",
    "import pandas as pd\n",
    "\n",
    "def balance_and_augment(df):\n",
    "    # Augment the dataset\n",
    "    smote_nc = SMOTENC(categorical_features=[1], random_state=0)\n",
    "    X_res, y_res = smote_nc.fit_resample(df[['ecg_id', 'filename_hr']].to_numpy(), df['label'])\n",
    "    df_balanced = pd.DataFrame(X_res, columns=['ecg_id', 'filename_hr'])\n",
    "    df_balanced['label'] = y_res\n",
    "\n",
    "    return df_balanced"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0301b2a5-158e-44a7-9144-2d19827e4e51",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    ecg_id                filename_hr label\n",
      "0   4759.0  records500/04000/04759_hr  NORM\n",
      "1   9278.0  records500/09000/09278_hr  NORM\n",
      "2  13398.0  records500/13000/13398_hr  NORM\n",
      "3  11144.0  records500/11000/11144_hr  NORM\n",
      "4    399.0  records500/00000/00399_hr  NORM\n"
     ]
    }
   ],
   "source": [
    "# Collecting and labeling dataset\n",
    "df_labeled = collect_and_label('ptbxl_database.csv')\n",
    "\n",
    "# Balancing and augmenting the dataset\n",
    "df_balanced_and_augmented = balance_and_augment(data)\n",
    "\n",
    "# Printing the first few rows of the resulting DataFrame\n",
    "print(df_balanced_and_augmented.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "4af5267a-2e9c-4b9f-ab5b-4fb7b062677f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Label 'NORM' has 2232 records.\n",
      "Label 'IMI' has 2232 records.\n"
     ]
    }
   ],
   "source": [
    "label_counts = Counter(df_balanced_and_augmented['label'])\n",
    "for label, count in label_counts.items():\n",
    "    print(f\"Label '{label}' has {count} records.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7dc838c2-d918-475a-8a49-71e18cb1b7e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_balanced_and_augmented=df_balanced_and_augmented.sample(frac = 1 , ignore_index=True, random_state=123)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f29e13e1-0843-4275-be08-d9dfad67ce4c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ecg_id</th>\n",
       "      <th>filename_hr</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7781.0</td>\n",
       "      <td>records500/07000/07781_hr</td>\n",
       "      <td>IMI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4036.0</td>\n",
       "      <td>records500/04000/04036_hr</td>\n",
       "      <td>NORM</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8857.0</td>\n",
       "      <td>records500/08000/08857_hr</td>\n",
       "      <td>NORM</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>9380.0</td>\n",
       "      <td>records500/09000/09380_hr</td>\n",
       "      <td>IMI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>14816.0</td>\n",
       "      <td>records500/14000/14816_hr</td>\n",
       "      <td>NORM</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4459</th>\n",
       "      <td>18605.0</td>\n",
       "      <td>records500/18000/18605_hr</td>\n",
       "      <td>NORM</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4460</th>\n",
       "      <td>18949.0</td>\n",
       "      <td>records500/18000/18949_hr</td>\n",
       "      <td>IMI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4461</th>\n",
       "      <td>18242.0</td>\n",
       "      <td>records500/18000/18242_hr</td>\n",
       "      <td>NORM</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4462</th>\n",
       "      <td>15365.0</td>\n",
       "      <td>records500/15000/15365_hr</td>\n",
       "      <td>IMI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4463</th>\n",
       "      <td>4755.0</td>\n",
       "      <td>records500/04000/04755_hr</td>\n",
       "      <td>IMI</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>4464 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       ecg_id                filename_hr label\n",
       "0      7781.0  records500/07000/07781_hr   IMI\n",
       "1      4036.0  records500/04000/04036_hr  NORM\n",
       "2      8857.0  records500/08000/08857_hr  NORM\n",
       "3      9380.0  records500/09000/09380_hr   IMI\n",
       "4     14816.0  records500/14000/14816_hr  NORM\n",
       "...       ...                        ...   ...\n",
       "4459  18605.0  records500/18000/18605_hr  NORM\n",
       "4460  18949.0  records500/18000/18949_hr   IMI\n",
       "4461  18242.0  records500/18000/18242_hr  NORM\n",
       "4462  15365.0  records500/15000/15365_hr   IMI\n",
       "4463   4755.0  records500/04000/04755_hr   IMI\n",
       "\n",
       "[4464 rows x 3 columns]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_balanced_and_augmented"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a5aa783b-355c-42f4-a384-decbf186ab89",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Missing values:\n",
      "ecg_id         0\n",
      "filename_hr    0\n",
      "label          0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Identify missing values\n",
    "missing_values = df_balanced_and_augmented.isnull().sum()\n",
    "print(\"Missing values:\")\n",
    "print(missing_values)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "cc44cf91-8ee8-44b4-af51-17eac2d982ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ecg_id                           16073.0\n",
      "filename_hr    records500/16000/16073_hr\n",
      "label                               NORM\n",
      "Name: 200, dtype: object\n"
     ]
    }
   ],
   "source": [
    "# Display the first record from the DataFrame\n",
    "print(df_balanced_and_augmented.iloc[200])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "0a91c863-58a0-4c83-8803-cc07d5ebea6b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       ecg_id                filename_hr label  \\\n",
      "0      7781.0  records500/07000/07781_hr   IMI   \n",
      "1      4036.0  records500/04000/04036_hr  NORM   \n",
      "2      8857.0  records500/08000/08857_hr  NORM   \n",
      "3      9380.0  records500/09000/09380_hr   IMI   \n",
      "4     14816.0  records500/14000/14816_hr  NORM   \n",
      "...       ...                        ...   ...   \n",
      "4459  18605.0  records500/18000/18605_hr  NORM   \n",
      "4460  18949.0  records500/18000/18949_hr   IMI   \n",
      "4461  18242.0  records500/18000/18242_hr  NORM   \n",
      "4462  15365.0  records500/15000/15365_hr   IMI   \n",
      "4463   4755.0  records500/04000/04755_hr   IMI   \n",
      "\n",
      "                              filtered_lead_1_channel_0  \n",
      "0     [-0.05535706432148732, -0.055705980314489374, ...  \n",
      "1     [-0.017722321193429806, -0.017834025117279443,...  \n",
      "2     [-0.037288157163053365, -0.037523184698260166,...  \n",
      "3     [-0.02261151421908207, -0.0227540347633655, -0...  \n",
      "4     [-0.025732273877728324, -0.02587018510813184, ...  \n",
      "...                                                 ...  \n",
      "4459  [0.30157904947073444, 0.3032310780196683, 0.30...  \n",
      "4460  [-0.040269239174407515, -0.04043204692095402, ...  \n",
      "4461  [-0.029813800094895503, -0.029904637258869216,...  \n",
      "4462  [0.06885256368921396, 0.06923798294530313, 0.0...  \n",
      "4463  [-0.0015543383073876277, -0.001564135313436100...  \n",
      "\n",
      "[4464 rows x 4 columns]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy.signal import butter, filtfilt\n",
    "import wfdb\n",
    "\n",
    "# Apply Butterworth high-pass filter\n",
    "def apply_highpass_filter(signal, lowcut, fs, order=1):\n",
    "    nyq = 0.5 * fs\n",
    "    low = lowcut / nyq\n",
    "    b, a = butter(order, low, btype='high')\n",
    "    return filtfilt(b, a, signal)\n",
    "\n",
    "# Function to apply high-pass filter to lead 1 at channel 0\n",
    "def apply_highpass_filter_to_lead1(file_path):\n",
    "    # Load ECG record\n",
    "    record = wfdb.rdrecord(file_path)\n",
    "\n",
    "    # Extract lead 1 signal (assuming it's at channel 0)\n",
    "    lead_1_signal = record.p_signal[:, 0]\n",
    "\n",
    "    # Sampling frequency\n",
    "    fs = record.fs\n",
    "\n",
    "    # Apply high-pass filter\n",
    "    lowcut = 0.5  # Adjust cutoff frequency as needed\n",
    "    filtered_lead_1_signal = apply_highpass_filter(lead_1_signal, lowcut, fs)\n",
    "\n",
    "    return filtered_lead_1_signal\n",
    "\n",
    "# Assuming 'filename_hr' contains file paths to ECG signals\n",
    "file_paths = df_balanced_and_augmented['filename_hr']\n",
    "\n",
    "# Apply high-pass filter to lead 1 at channel 0 for each file\n",
    "filtered_signals = []\n",
    "for file_path in file_paths:\n",
    "    filtered_signal = apply_highpass_filter_to_lead1(file_path)\n",
    "    filtered_signals.append(filtered_signal)\n",
    "\n",
    "# Add filtered signals to DataFrame\n",
    "df_balanced_and_augmented['filtered_lead_1_channel_0'] = filtered_signals\n",
    "\n",
    "# Display DataFrame with filtered signals\n",
    "print(df_balanced_and_augmented)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "258ef4ca-58e4-444c-b3a9-3d6ecb4fd625",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from tensorflow import data \n",
    "import wfdb as sig\n",
    "import keras\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Activation, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D\n",
    "from keras import regularizers\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "02d48ae8-401b-43a0-8db0-c8aefd1cd657",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess(dat):\n",
    "    data_dir = list(dat['filename_hr'])\n",
    "    data_signal = map(read_signal , data_dir)\n",
    "    data_signal = list(data_signal)\n",
    "    data_signal = np.array(data_signal)\n",
    "    data_dict = {'NORM' : 0 , 'IMI': 1  }\n",
    "    encoded_label = dat['label'].map(data_dict)\n",
    "    return np.array(data_signal)  , np.array(encoded_label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "9b1d7580-3bed-498e-b5b2-68f0047af84d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train size: 3124\n",
      "Validation size: 670\n",
      "Test size: 670\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Splitting data into train and (validation + test) sets\n",
    "train, val_test = train_test_split(df_balanced_and_augmented, train_size=0.7, random_state=1002)\n",
    "\n",
    "# Splitting (validation + test) into validation and test sets\n",
    "validation, test = train_test_split(val_test, test_size=0.5, random_state=1002)\n",
    "\n",
    "# Printing the sizes of train, validation, and test sets\n",
    "print(\"Train size:\", len(train))\n",
    "print(\"Validation size:\", len(validation))\n",
    "print(\"Test size:\", len(test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d96cf35c-fbe9-4d14-8bc4-9e1ff5c2c0d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3124, 4)\n",
      "(670, 4)\n",
      "(670, 4)\n"
     ]
    }
   ],
   "source": [
    "print(train.shape)\n",
    "print(validation.shape)\n",
    "print(test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "cc974c5c-6b08-4bfd-a316-0725beb2327f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def read_signal(record):\n",
    "    tes = sig.rdrecord(record,sampfrom=0 , sampto=5000)\n",
    "    signal = tes.__dict__['p_signal'][::,0]\n",
    "    return signal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "2fd072f0-ea15-4ece-96c7-1a0655b90b09",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train , y_train = preprocess(train)\n",
    "X_valid , y_valid = preprocess(validation)\n",
    "X_test  , y_test  = preprocess(test) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "09e7e807-fdc8-419c-b07a-b06692187aa5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3124, 5000)\n",
      "(3124,)\n",
      "(670, 5000)\n",
      "(670,)\n",
      "(670, 5000)\n",
      "(670,)\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape)\n",
    "print(y_train.shape)\n",
    "print(X_valid.shape)\n",
    "print(y_valid.shape)\n",
    "print(X_test.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "dfef9da2-4c6f-4c58-9be2-6572ab8d0fd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reshape input data to add the timestep dimension\n",
    "X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)\n",
    "X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)\n",
    "X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "6a044ba8-03c8-4de8-b70b-2410bb9664b3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3124, 5000, 1)\n",
      "(3124,)\n",
      "(670, 5000, 1)\n",
      "(670,)\n",
      "(670, 5000, 1)\n",
      "(670,)\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape)\n",
    "print(y_train.shape)\n",
    "print(X_valid.shape)\n",
    "print(y_valid.shape)\n",
    "print(X_test.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "e4fa72d9-3654-42c9-b95f-5a33a75d334b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential_4\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential_4\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv1d_8 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv1D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4996</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)            │             <span style=\"color: #00af00; text-decoration-color: #00af00\">192</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ batch_normalization_2                │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4996</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)            │             <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)                 │                             │                 │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling1d_8 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling1D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2498</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)            │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2498</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)            │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv1d_9 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv1D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2496</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)            │           <span style=\"color: #00af00; text-decoration-color: #00af00\">6,208</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ batch_normalization_3                │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2496</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)            │             <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)                 │                             │                 │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling1d_9 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling1D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1248</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)            │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_5 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1248</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)            │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ lstm_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">LSTM</span>)                        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1248</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)            │          <span style=\"color: #00af00; text-decoration-color: #00af00\">33,024</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_6 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1248</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)            │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">79872</span>)               │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_6 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)                 │      <span style=\"color: #00af00; text-decoration-color: #00af00\">10,223,744</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_7 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)                 │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_7 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>)                   │             <span style=\"color: #00af00; text-decoration-color: #00af00\">258</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv1d_8 (\u001b[38;5;33mConv1D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4996\u001b[0m, \u001b[38;5;34m32\u001b[0m)            │             \u001b[38;5;34m192\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ batch_normalization_2                │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4996\u001b[0m, \u001b[38;5;34m32\u001b[0m)            │             \u001b[38;5;34m128\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)                 │                             │                 │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling1d_8 (\u001b[38;5;33mMaxPooling1D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2498\u001b[0m, \u001b[38;5;34m32\u001b[0m)            │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_4 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2498\u001b[0m, \u001b[38;5;34m32\u001b[0m)            │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv1d_9 (\u001b[38;5;33mConv1D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2496\u001b[0m, \u001b[38;5;34m64\u001b[0m)            │           \u001b[38;5;34m6,208\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ batch_normalization_3                │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2496\u001b[0m, \u001b[38;5;34m64\u001b[0m)            │             \u001b[38;5;34m256\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)                 │                             │                 │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling1d_9 (\u001b[38;5;33mMaxPooling1D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1248\u001b[0m, \u001b[38;5;34m64\u001b[0m)            │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_5 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1248\u001b[0m, \u001b[38;5;34m64\u001b[0m)            │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ lstm_3 (\u001b[38;5;33mLSTM\u001b[0m)                        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1248\u001b[0m, \u001b[38;5;34m64\u001b[0m)            │          \u001b[38;5;34m33,024\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_6 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1248\u001b[0m, \u001b[38;5;34m64\u001b[0m)            │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten_3 (\u001b[38;5;33mFlatten\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m79872\u001b[0m)               │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_6 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m)                 │      \u001b[38;5;34m10,223,744\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_7 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m)                 │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_7 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m)                   │             \u001b[38;5;34m258\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">10,263,810</span> (39.15 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m10,263,810\u001b[0m (39.15 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">10,263,618</span> (39.15 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m10,263,618\u001b[0m (39.15 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">192</span> (768.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m192\u001b[0m (768.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 490ms/step - loss: 0.7375 - sparse_categorical_accuracy: 0.5431\n",
      "Epoch 1: val_sparse_categorical_accuracy improved from -inf to 0.52239, saving model to best_model.keras\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m56s\u001b[0m 537ms/step - loss: 0.7370 - sparse_categorical_accuracy: 0.5437 - val_loss: 0.8431 - val_sparse_categorical_accuracy: 0.5224\n",
      "Epoch 2/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 469ms/step - loss: 0.5505 - sparse_categorical_accuracy: 0.7213\n",
      "Epoch 2: val_sparse_categorical_accuracy improved from 0.52239 to 0.52537, saving model to best_model.keras\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m50s\u001b[0m 512ms/step - loss: 0.5506 - sparse_categorical_accuracy: 0.7213 - val_loss: 0.7470 - val_sparse_categorical_accuracy: 0.5254\n",
      "Epoch 3/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 464ms/step - loss: 0.4404 - sparse_categorical_accuracy: 0.7992\n",
      "Epoch 3: val_sparse_categorical_accuracy did not improve from 0.52537\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 490ms/step - loss: 0.4404 - sparse_categorical_accuracy: 0.7992 - val_loss: 1.9780 - val_sparse_categorical_accuracy: 0.4776\n",
      "Epoch 4/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 465ms/step - loss: 0.3429 - sparse_categorical_accuracy: 0.8597\n",
      "Epoch 4: val_sparse_categorical_accuracy did not improve from 0.52537\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 490ms/step - loss: 0.3430 - sparse_categorical_accuracy: 0.8597 - val_loss: 3.5473 - val_sparse_categorical_accuracy: 0.4776\n",
      "Epoch 5/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 468ms/step - loss: 0.2682 - sparse_categorical_accuracy: 0.9024\n",
      "Epoch 5: val_sparse_categorical_accuracy did not improve from 0.52537\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 493ms/step - loss: 0.2682 - sparse_categorical_accuracy: 0.9024 - val_loss: 5.2003 - val_sparse_categorical_accuracy: 0.4776\n",
      "Epoch 6/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 470ms/step - loss: 0.1814 - sparse_categorical_accuracy: 0.9440\n",
      "Epoch 6: val_sparse_categorical_accuracy did not improve from 0.52537\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 495ms/step - loss: 0.1815 - sparse_categorical_accuracy: 0.9439 - val_loss: 3.9559 - val_sparse_categorical_accuracy: 0.4776\n",
      "Epoch 7/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 472ms/step - loss: 0.1396 - sparse_categorical_accuracy: 0.9572\n",
      "Epoch 7: val_sparse_categorical_accuracy improved from 0.52537 to 0.52687, saving model to best_model.keras\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m50s\u001b[0m 515ms/step - loss: 0.1397 - sparse_categorical_accuracy: 0.9571 - val_loss: 2.1454 - val_sparse_categorical_accuracy: 0.5269\n",
      "Epoch 8/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 475ms/step - loss: 0.1071 - sparse_categorical_accuracy: 0.9740\n",
      "Epoch 8: val_sparse_categorical_accuracy improved from 0.52687 to 0.62239, saving model to best_model.keras\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m51s\u001b[0m 518ms/step - loss: 0.1071 - sparse_categorical_accuracy: 0.9739 - val_loss: 1.4875 - val_sparse_categorical_accuracy: 0.6224\n",
      "Epoch 9/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 469ms/step - loss: 0.0777 - sparse_categorical_accuracy: 0.9867\n",
      "Epoch 9: val_sparse_categorical_accuracy improved from 0.62239 to 0.63731, saving model to best_model.keras\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m52s\u001b[0m 534ms/step - loss: 0.0777 - sparse_categorical_accuracy: 0.9866 - val_loss: 1.3361 - val_sparse_categorical_accuracy: 0.6373\n",
      "Epoch 10/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 469ms/step - loss: 0.0540 - sparse_categorical_accuracy: 0.9889\n",
      "Epoch 10: val_sparse_categorical_accuracy improved from 0.63731 to 0.67164, saving model to best_model.keras\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m52s\u001b[0m 532ms/step - loss: 0.0540 - sparse_categorical_accuracy: 0.9889 - val_loss: 1.2229 - val_sparse_categorical_accuracy: 0.6716\n",
      "Epoch 11/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 469ms/step - loss: 0.0480 - sparse_categorical_accuracy: 0.9909\n",
      "Epoch 11: val_sparse_categorical_accuracy did not improve from 0.67164\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 495ms/step - loss: 0.0480 - sparse_categorical_accuracy: 0.9909 - val_loss: 1.4618 - val_sparse_categorical_accuracy: 0.6358\n",
      "Epoch 12/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 469ms/step - loss: 0.0424 - sparse_categorical_accuracy: 0.9894\n",
      "Epoch 12: val_sparse_categorical_accuracy did not improve from 0.67164\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m49s\u001b[0m 495ms/step - loss: 0.0424 - sparse_categorical_accuracy: 0.9894 - val_loss: 1.2750 - val_sparse_categorical_accuracy: 0.6612\n",
      "Epoch 13/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 468ms/step - loss: 0.0313 - sparse_categorical_accuracy: 0.9943\n",
      "Epoch 13: val_sparse_categorical_accuracy did not improve from 0.67164\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 495ms/step - loss: 0.0313 - sparse_categorical_accuracy: 0.9943 - val_loss: 1.5281 - val_sparse_categorical_accuracy: 0.6418\n",
      "Epoch 14/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 471ms/step - loss: 0.0237 - sparse_categorical_accuracy: 0.9960\n",
      "Epoch 14: val_sparse_categorical_accuracy did not improve from 0.67164\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m49s\u001b[0m 496ms/step - loss: 0.0238 - sparse_categorical_accuracy: 0.9959 - val_loss: 1.4295 - val_sparse_categorical_accuracy: 0.6507\n",
      "Epoch 15/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 465ms/step - loss: 0.0262 - sparse_categorical_accuracy: 0.9952\n",
      "Epoch 15: val_sparse_categorical_accuracy did not improve from 0.67164\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 490ms/step - loss: 0.0261 - sparse_categorical_accuracy: 0.9952 - val_loss: 1.4632 - val_sparse_categorical_accuracy: 0.6672\n",
      "Epoch 16/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 460ms/step - loss: 0.0158 - sparse_categorical_accuracy: 0.9980\n",
      "Epoch 16: val_sparse_categorical_accuracy did not improve from 0.67164\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m47s\u001b[0m 484ms/step - loss: 0.0159 - sparse_categorical_accuracy: 0.9979 - val_loss: 1.7459 - val_sparse_categorical_accuracy: 0.6313\n",
      "Epoch 17/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 459ms/step - loss: 0.0140 - sparse_categorical_accuracy: 0.9989\n",
      "Epoch 17: val_sparse_categorical_accuracy improved from 0.67164 to 0.67761, saving model to best_model.keras\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m50s\u001b[0m 508ms/step - loss: 0.0140 - sparse_categorical_accuracy: 0.9989 - val_loss: 1.5160 - val_sparse_categorical_accuracy: 0.6776\n",
      "Epoch 18/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 469ms/step - loss: 0.0141 - sparse_categorical_accuracy: 0.9978\n",
      "Epoch 18: val_sparse_categorical_accuracy did not improve from 0.67761\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 494ms/step - loss: 0.0141 - sparse_categorical_accuracy: 0.9978 - val_loss: 1.6059 - val_sparse_categorical_accuracy: 0.6448\n",
      "Epoch 19/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 469ms/step - loss: 0.0106 - sparse_categorical_accuracy: 0.9987\n",
      "Epoch 19: val_sparse_categorical_accuracy did not improve from 0.67761\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 494ms/step - loss: 0.0106 - sparse_categorical_accuracy: 0.9987 - val_loss: 1.5445 - val_sparse_categorical_accuracy: 0.6731\n",
      "Epoch 20/20\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 469ms/step - loss: 0.0106 - sparse_categorical_accuracy: 0.9989\n",
      "Epoch 20: val_sparse_categorical_accuracy did not improve from 0.67761\n",
      "\u001b[1m98/98\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m48s\u001b[0m 495ms/step - loss: 0.0106 - sparse_categorical_accuracy: 0.9989 - val_loss: 1.6070 - val_sparse_categorical_accuracy: 0.6687\n",
      "Test Loss: 1.260242223739624, Test Accuracy: 0.7104477882385254\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, BatchNormalization\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.losses import SparseCategoricalCrossentropy\n",
    "from tensorflow.keras.metrics import SparseCategoricalAccuracy\n",
    "from tensorflow.keras.callbacks import ModelCheckpoint\n",
    "\n",
    "# Function to build the CRNN model\n",
    "def build_crnn_model(input_shape, num_classes):\n",
    "    model = Sequential()\n",
    "    \n",
    "    # CNN layers\n",
    "    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(MaxPooling1D(pool_size=2))\n",
    "    model.add(Dropout(0.2))\n",
    "    \n",
    "    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(MaxPooling1D(pool_size=2))\n",
    "    model.add(Dropout(0.2))\n",
    "    \n",
    "    # LSTM layer\n",
    "    model.add(LSTM(64, return_sequences=True))\n",
    "    model.add(Dropout(0.2))\n",
    "    \n",
    "    # Flatten before dense layer\n",
    "    model.add(Flatten())\n",
    "    \n",
    "    # Dense layers\n",
    "    model.add(Dense(128, activation='relu'))\n",
    "    model.add(Dropout(0.2))\n",
    "    model.add(Dense(num_classes, activation='softmax'))\n",
    "    \n",
    "    return model\n",
    "\n",
    "# Define input shape and number of classes\n",
    "input_shape = X_train.shape[1:]\n",
    "num_classes = 2  # Two classes: normal and abnormal\n",
    "\n",
    "# Build the CRNN model\n",
    "crnn_model = build_crnn_model(input_shape, num_classes)\n",
    "\n",
    "# Compile the model with a lower learning rate\n",
    "crnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])\n",
    "\n",
    "# Print model summary\n",
    "crnn_model.summary()\n",
    "\n",
    "# Define the ModelCheckpoint callback\n",
    "checkpoint = ModelCheckpoint('best_model.keras', monitor='val_sparse_categorical_accuracy', save_best_only=True, mode='max', verbose=1)\n",
    "\n",
    "# Train the model\n",
    "batch_size = 32\n",
    "epochs = 20  # Increased number of epochs\n",
    "\n",
    "history = crnn_model.fit(X_train, y_train, \n",
    "                         batch_size=batch_size, \n",
    "                         epochs=epochs, \n",
    "                         validation_data=(X_valid, y_valid),\n",
    "                         callbacks=[checkpoint])\n",
    "\n",
    "# Evaluate the model on the test set\n",
    "test_loss, test_accuracy = crnn_model.evaluate(X_test, y_test, verbose=0)\n",
    "print(f\"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "9494f1d7-d46d-4e87-8956-48b941cee50c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 167ms/step\n"
     ]
    }
   ],
   "source": [
    "# Predict probabilities on the test set\n",
    "y_pred_proba = crnn_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "fb5738b3-e89d-4da2-88a0-4c537c630c90",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Threshold: 0.8188188188188188\n",
      "Best Metric Sum (TP + TN - FP - FN): 308\n",
      "True Positives with Optimal Threshold: 254\n",
      "True Negatives with Optimal Threshold: 235\n",
      "False Positives with Optimal Threshold: 90\n",
      "False Negatives with Optimal Threshold: 91\n",
      "Precision with Optimal Threshold: 0.7383720930232558\n",
      "F1 Score with Optimal Threshold: 0.7373004354136431\n",
      "Sensitivity (Recall) with Optimal Threshold: 0.736231884057971\n",
      "Specificity with Optimal Threshold: 0.7230769230769231\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.metrics import confusion_matrix, precision_score, f1_score\n",
    "\n",
    "# Define the threshold search range\n",
    "thresholds = np.linspace(0, 1, 1000)\n",
    "\n",
    "# Initialize variables to store best threshold and corresponding metrics\n",
    "best_threshold = None\n",
    "best_metric_sum = float('-inf')  # We want to maximize this sum\n",
    "\n",
    "# Loop through each threshold and calculate corresponding metrics\n",
    "for threshold in thresholds:\n",
    "    y_pred_thresholded = (y_pred_proba[:, 1] > threshold).astype(int)\n",
    "    cm = confusion_matrix(y_test, y_pred_thresholded)\n",
    "    TP = cm[1, 1]\n",
    "    TN = cm[0, 0]\n",
    "    FP = cm[0, 1]\n",
    "    FN = cm[1, 0]\n",
    "    \n",
    "    # Calculate the sum of true positives and true negatives while minimizing false positives and false negatives\n",
    "    metric_sum = TP + TN - FP - FN\n",
    "    \n",
    "    # Update best threshold and best metric sum if a better threshold is found\n",
    "    if metric_sum > best_metric_sum:\n",
    "        best_metric_sum = metric_sum\n",
    "        best_threshold = threshold\n",
    "\n",
    "# Print the best threshold and corresponding metrics\n",
    "print(\"Best Threshold:\", best_threshold)\n",
    "print(\"Best Metric Sum (TP + TN - FP - FN):\", best_metric_sum)\n",
    "\n",
    "# Convert predicted probabilities to class labels using the optimal threshold\n",
    "y_pred_optimal = (y_pred_proba[:, 1] > best_threshold).astype(int)\n",
    "\n",
    "# Compute confusion matrix using the optimal threshold\n",
    "cm_optimal = confusion_matrix(y_test, y_pred_optimal)\n",
    "\n",
    "# Extract TP, TN, FP, FN using the optimal threshold\n",
    "TP_optimal = cm_optimal[1, 1]\n",
    "TN_optimal = cm_optimal[0, 0]\n",
    "FP_optimal = cm_optimal[0, 1]\n",
    "FN_optimal = cm_optimal[1, 0]\n",
    "\n",
    "# Print the number of TP, TN, FP, FN using the optimal threshold\n",
    "print(\"True Positives with Optimal Threshold:\", TP_optimal)\n",
    "print(\"True Negatives with Optimal Threshold:\", TN_optimal)\n",
    "print(\"False Positives with Optimal Threshold:\", FP_optimal)\n",
    "print(\"False Negatives with Optimal Threshold:\", FN_optimal)\n",
    "\n",
    "# Compute precision, F1 score, sensitivity (recall), and specificity\n",
    "precision_optimal = precision_score(y_test, y_pred_optimal)\n",
    "f1_optimal = f1_score(y_test, y_pred_optimal)\n",
    "sensitivity_optimal = TP_optimal / (TP_optimal + FN_optimal)\n",
    "specificity_optimal = TN_optimal / (TN_optimal + FP_optimal)\n",
    "\n",
    "# Print precision, F1 score, sensitivity (recall), and specificity\n",
    "print(\"Precision with Optimal Threshold:\", precision_optimal)\n",
    "print(\"F1 Score with Optimal Threshold:\", f1_optimal)\n",
    "print(\"Sensitivity (Recall) with Optimal Threshold:\", sensitivity_optimal)\n",
    "print(\"Specificity with Optimal Threshold:\", specificity_optimal)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "490d1085-8962-4c19-89b4-ffa8bc834b50",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix with Optimal Threshold:\n",
      "[[235  90]\n",
      " [ 91 254]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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1b59e73wwgt677339Ntvvyk+Pl7x8fGaM2eOJKlv3776/ffftX79epUvX14jRozQ6dOnbcZxc3PTjBkzVLt2bf3000967LHH9NRTT+nNN99Us2bNNH36dI0bN04pKSmSpIoVKxaqNSYmRg888IAyMzOtx1etWqXz58+ra9eukqSEhAR98MEHSkxM1A033KCNGzfqoYceUpUqVdS6desrfgbZ2dnKzs62vs7IyLjGTxOu4JGON0mSkl7qYrN/0Ovr9EFyirJz89U8opri722gSj6eOp1+QZv2nlTbpxfpt/RLK5/k5hXokY43a/KAZrLIosMn0zX6vS16b/V+098PAOdIT0/XuOfG6NdfflFgYKC6dO2uCZNeVPny5SVJT458SuezshT/6GClpaWpWfMWWrJsJWt4A6WYxTAM469Pc45+/fopLS1NixcvVmRkpCIiIvTuu+9q8eLF6tq1qwzD0MCBA1WuXDm99dZb1us2bdqk1q1bKysrS0ePHlX9+vW1Y8cO3XbbbZKkQ4cO6YYbbtC0adM0bNiwK977008/1ZAhQ/T7779LujTHe9iwYUpLS7M5r1atWho2bJiGDRumvLw8VatWTVOnTlWfPn0kXUrBCwoK9OGHHyo7O1uBgYFas2aNIiP//5/6Bg4cqPPnz2vhwoVXrGX8+PGaMGFCof2eHabKUt67yJ8ngLLt7GeFn1YK4J8rIyNDQZX9lZ6eLj8/v5IuRxkZGfL391fYY5/IzdO503QLss/r5zcfKDXvvahKfKrJZa+88ormzZun/fttE7vvvvtOc+fOVcWKFa1bdHS0CgoKdOTIEaWkpMjd3V233nqr9Zrw8HBVqlTJZpw1a9borrvuUvXq1eXr66s+ffrojz/+0Pnz54tco7u7u3r06KEFCxZIkrKysvTFF18oJiZG0qWG//z587r77rtt6p0/f74OH7b/GO8xY8YoPT3duh0/frzINQEAAMA1lPhUk8tatWql6OhojRkzRv369bPuz8zM1COPPKKhQ4cWuqZmzZo6cODAX4599OhR3XPPPXr00Uf14osvKjAwUJs2bdKAAQOUk5NTrC9PxsTEqHXr1jp9+rSSkpLk7e2t9u3bW2uVpOXLl6t69eo213l6etod09PT86rHAQAAXIUZq4646qompabxlqSXX35ZjRo1Ut26//8EvltvvVX79u1TeHj4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AADg8oqdeC9dulTz589XmzZt1L9/f7Vs2VLh4eEKCwvTggULFBMT44w6AQAA4AL4cqV9xU68z5w5o+uvv17SpfncZ86ckSS1aNFCGzdudGx1AAAAQBlR7Mb7+uuv15EjRyRJ9erV08cffyzpUhIeEBDg0OIAAADgWniAjn3Fbrz79++v7777TpL09NNPa9asWfLy8tLw4cM1atQohxcIAAAAlAXFnuM9fPhw6/+OiorSjz/+qF27dik8PFy33HKLQ4sDAACAa2GOt31/ax1vSQoLC1NYWJgjagEAAADKrCI13jNmzCjygEOHDr3mYgAAAODaWMfbviI13tOmTSvSYBaLhcbbgX5+/2GeBArAqtLt8SVdAoBSxMjP+euTUKoUqfG+vIoJAAAAcDVuuobVO67hHq7IVesGAAAAXMrf/nIlAAAAcBlzvO0j8QYAAABMQOINAAAAh7FYJDfW8b4iEm8AAADABNfUeH/11Vd66KGHFBkZqV9//VWS9P7772vTpk0OLQ4AAACuxc1izuaKit14f/bZZ4qOjpa3t7e+/fZbZWdnS5LS09P10ksvObxAAAAAoCwoduP9wgsvKDExUW+//bbKly9v3d+8eXN98803Di0OAAAAruXyqibO3lxRsRvvlJQUtWrVqtB+f39/paWlOaImAAAAoMwpduMdHBysQ4cOFdq/adMmXX/99Q4pCgAAAK6JOd72FbvxHjRokJ544glt375dFotFJ06c0IIFCzRy5Eg9+uijzqgRAAAAcHnFXsf76aefVkFBge666y6dP39erVq1kqenp0aOHKnHH3/cGTUCAADARVgszl9n20WneBe/8bZYLHr22Wc1atQoHTp0SJmZmYqIiFDFihWdUR8AAABQJlzzkys9PDwUERHhyFoAAADg4twsFrk5OZJ29vjOUuzGu23btlddwiU5OflvFQQAAACURcVuvBs1amTzOjc3V7t379YPP/yg2NhYR9UFAAAAF+Sma3w0ejHv4YqK3XhPmzbtivvHjx+vzMzMv10QAAAAUBY57BeGhx56SO+9956jhgMAAIALuryqibM3V+Swxnvr1q3y8vJy1HAAAABAmVLsqSbdunWzeW0Yhk6ePKmdO3dq7NixDisMAAAArsdNJqxqIteMvIvdePv7+9u8dnNzU926dTVx4kS1a9fOYYUBAAAAZUmxGu/8/Hz1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6iyq0aNH68Ybb1SFChV0/fXXa+zYscrNzS103ltvvaXQ0FBVqFBBPXr0UHp6us3xd955R/Xr15eXl5fq1aunN998s9i1AEBZReMN4B/H29tbOTk51tdr165VSkqKkpKStGzZMuXm5io6Olq+vr766quvtHnzZlWsWFHt27e3XjdlyhTNnTtX7733njZt2qQzZ85o0aJFV71v37599Z///EczZszQ/v379dZbb6lixYoKDQ3VZ599JklKSUnRyZMn9frrr0uSEhISNH/+fCUmJmrv3r0aPny4HnroIW3YsEHSpV8QunXrps6dO2v37t0aOHCgnn766WJ/Jr6+vpo7d6727dun119/XW+//bamTZtmc86hQ4f08ccfa+nSpVq5cqW+/fZbPfbYY9bjCxYs0Lhx4/Tiiy9q//79eumllzR27FjNmzev2PUAQJlkAEAZFhsba3Tp0sUwDMMoKCgwkpKSDE9PT2PkyJHW40FBQUZ2drb1mvfff9+oW7euUVBQYN2XnZ1teHt7G6tWrTIMwzCqVatmTJ482Xo8NzfXqFGjhvVehmEYrVu3Np544gnDMAwjJSXFkGQkJSVdsc5169YZkoyzZ89a9128eNGoUKGCsWXLFptzBwwYYDz44IOGYRjGmDFjjIiICJvjo0ePLjTW/5JkLFq0yO7xV1991WjSpIn19fPPP2+UK1fO+OWXX6z7vvzyS8PNzc04efKkYRiGUadOHWPhwoU240yaNMmIjIw0DMMwjhw5Ykgyvv32W7v3BYCyjDneAMq8ZcuWqWLFisrNzVVBQYF69+6t8ePHW483aNDAZl73d999p0OHDsnX19dmnIsXL+rw4cNKT0/XyZMn1bRpU+sxd3d33XbbbYWmm1y2e/dulStXTq1bty5y3YcOHdL58+d199132+zPyclR48aNJUn79++3qUOSIiMji3yPyz766CPNmDFDhw8fVmZmpvLy8uTn52dzTs2aNVW9enWb+xQUFCglJUW+vr46fPiwBgwYoEGDBlnPycvLk7+/f7HrAYCyiMYbQJnXtm1bzZ49Wx4eHgoJCZG7u+2PPh8fH5vXmZmZatKkiRYsWFBorCpVqlxTDd7e3sW+JjMzU5K0fPlym4ZXujRv3VG2bt2qmJgYTZgwQdHR0fL399eHH36oKVOmFLvWt99+u9AvAuXKlXNYrQDgymi8AZR5Pj4+Cg8PL/L5t956qz766CNVrVq1UOp7WbVq1bR9+3a1atVK0qVkd9euXbr11luveH6DBg1UUFCgDRs2KCoqqtDxy4l7fn6+dV9ERIQ8PT117Ngxu0l5/fr1rV8UvWzbtm1//Sb/ZMuWLQoLC9Ozzz5r3ffzzz8XOu/YsWM6ceKEQkJCrPdxc3NT3bp1FRQUpJCQEP3000+KiYkp1v0B4J+CL1cCwP+IiYnRddddpy5duuirr77SkSNHtH79eg0dOlS//PKLJOmJJ57Qyy+/rMWLF+vHH3/UY489dtU1uGvVqqXY2Fg9/PDDWrx4sXXMjz/+WJIUFhYmi8WiZcuW6bffflNmZqZ8fX01cuRIDR8+XPPmzdPhw4f1zTffaObMmdYvLA4ZMkQHDx7UqFGjlJKSooULF2ru3LnFer833HCDjh07pg8//FCHDx/WjBkzrvhFUS8vL8XGxuq7777TV199paFDh6pHjx4KDg6WJE2YMEEJCQmaMWOGDhw4oD179mjOnDmaOnVqseoBgLKKxhsA/keFChW0ceNG1axZU926dVP9+vU1YMAAXbx40ZqAP/nkk+rTp49iY2MVGRkpX19fde3a9arjzp49W/fff78ee+wx1atXT4MGDVJWVpYkqXr16powYYKefvppBQUFKT4+XpI0adIkjR07VgkJCapfv77at2+v5cuXq3bt2pIuzbv+7LPPtHjxYjVs2FCJiYl66aWXivV+7733Xg0fPlzx8fFq1KiRtmzZorFjxxY6Lzw8XN26dVPHjh3Vrl073XLLLTbLBQ4cOFDvvPOO5syZowYNGqh169aaO3eutVYA+KezGPa+CQQAAADAYUi8AQAAABPQeAMAAAAmoPEGAAAATEDjDQAAAJiAxhsAAAAwAY03AAAAYAIabwAAAMAENN4AAACACWi8AQAAABPQeAMAAAAmoPEGAAAATPB/d/O0PFTypcQAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.metrics import confusion_matrix, precision_score, f1_score\n",
    "import matplotlib.pyplot as plt\n",
    "import itertools\n",
    "\n",
    "# Define function to plot confusion matrix\n",
    "def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, cm[i, j],\n",
    "                 horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('Predicted label')\n",
    "\n",
    "# Assuming you have already computed the optimal threshold metrics\n",
    "best_threshold = 0.8188188188188188\n",
    "y_pred_optimal = (y_pred_proba[:, 1] > best_threshold).astype(int)  # Assuming y_pred_proba is available\n",
    "\n",
    "# Compute confusion matrix using the optimal threshold\n",
    "cm_optimal = confusion_matrix(y_test, y_pred_optimal)\n",
    "\n",
    "# Define class labels\n",
    "classes = ['Negative', 'Positive']  # Replace with your actual class labels\n",
    "\n",
    "# Print confusion matrix details\n",
    "print(\"Confusion Matrix with Optimal Threshold:\")\n",
    "print(cm_optimal)\n",
    "\n",
    "# Plot confusion matrix for optimal threshold\n",
    "plt.figure(figsize=(8, 6))\n",
    "plot_confusion_matrix(cm_optimal, classes)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "e242487a-64ae-44fa-a01b-dd04b870b6cf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "# Compute ROC curve and ROC area for each class\n",
    "fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])\n",
    "roc_auc = auc(fpr, tpr)\n",
    "\n",
    "# Plot ROC curve\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))\n",
    "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver Operating Characteristic (ROC)')\n",
    "plt.legend(loc='lower right')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "0fea07cc-920c-481f-8016-080e64a34f08",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m21/21\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 157ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAArMAAAK9CAYAAAA37eRrAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/H5lhTAAAACXBIWXMAAA9hAAAPYQGoP6dpAACAEUlEQVR4nO3dd3gU1dvG8XsT0gmhhR4IvUivLyAdBAtSVKqCiKh0QRCQEkAFhB9NxYZKUZSioCgIAooUEenSO9I7JNQkZM/7R2TDkkI2JNls8v1c117snDkz8+xuonfOzpyxGGOMAAAAABfk5uwCAAAAgKQizAIAAMBlEWYBAADgsgizAAAAcFmEWQAAALgswiwAAABcFmEWAAAALoswCwAAAJdFmAUAAIDLIswC6VBwcLBefPFFZ5eR4dSvX1/169d3dhkPNHLkSFksFl28eNHZpaQ5FotFI0eOTJZ9HTt2TBaLRTNnzkyW/UnS33//LU9PT/3777/Jts/k1q5dO7Vp08bZZSADIcwCDpo5c6YsFovtkSlTJuXPn18vvviiTp065ezy0rQbN27o7bffVvny5eXr66uAgADVqVNHs2fPlqvcWXvPnj0aOXKkjh075uxSYomKitKMGTNUv359Zc+eXV5eXgoODlaXLl20efNmZ5eXLL755htNmTLF2WXYSc2ahg4dqvbt26tQoUK2tvr169v9N8nHx0fly5fXlClTZLVa49zPpUuXNHDgQJUsWVLe3t7Knj27mjZtqp9//jneY4eFhWnUqFGqUKGCMmfOLB8fH5UtW1aDBg3S6dOnbf0GDRqk77//Xjt27Ei+Fw4kwGJc5f8gQBoxc+ZMdenSRaNHj1bhwoV1+/Zt/fXXX5o5c6aCg4O1a9cueXt7O7XG8PBwubm5ycPDw6l13OvcuXNq1KiR9u7dq3bt2qlevXq6ffu2vv/+e61Zs0Zt27bVnDlz5O7u7uxSE/Tdd9/pueee0++//x5rFDYiIkKS5Onpmep13bp1S61bt9ayZctUt25dNW/eXNmzZ9exY8c0f/58HThwQMePH1eBAgU0cuRIjRo1ShcuXFDOnDlTvdaH8dRTT2nXrl0p9sfE7du3lSlTJmXKlOmhazLGKDw8XB4eHsnyc719+3ZVqlRJf/75p2rWrGlrr1+/vg4fPqyxY8dKki5evKhvvvlGmzZt0ltvvaV3333Xbj/79+9Xo0aNdOHCBXXp0kVVq1bV1atXNWfOHG3fvl0DBgzQhAkT7LY5cuSIGjdurOPHj+u5557To48+Kk9PT/3zzz/69ttvlT17dh04cMDWv0aNGipZsqRmz5790K8beCADwCEzZswwksymTZvs2gcNGmQkmXnz5jmpMue6deuWiYqKind906ZNjZubm/nxxx9jrRswYICRZMaNG5eSJcbp+vXrDvVfsGCBkWR+//33lCkoiXr27GkkmcmTJ8dad+fOHTNhwgRz4sQJY4wxISEhRpK5cOFCitVjtVrNzZs3k32/Tz75pClUqFCy7jMqKsrcunUrydunRE1x6dOnjylYsKCxWq127fXq1TOPPPKIXdutW7dMoUKFjL+/v7lz546tPSIiwpQtW9b4+vqav/76y26bO3fumLZt2xpJZu7cubb2yMhIU6FCBePr62vWrl0bq67Q0FDz1ltv2bX973//M35+fubatWtJfr1AYhFmAQfFF2Z//vlnI8mMGTPGrn3v3r3mmWeeMdmyZTNeXl6mSpUqcQa6K1eumNdff90UKlTIeHp6mvz585sXXnjBLnDcvn3bjBgxwhQtWtR4enqaAgUKmIEDB5rbt2/b7atQoUKmc+fOxhhjNm3aZCSZmTNnxjrmsmXLjCTz008/2dpOnjxpunTpYnLlymU8PT1NmTJlzBdffGG33e+//24kmW+//dYMHTrU5MuXz1gsFnPlypU437MNGzYYSeall16Kc31kZKQpXry4yZYtmy0AHT161EgyEyZMMJMmTTIFCxY03t7epm7dumbnzp2x9pGY9/nuZ7d69WrTvXt3ExgYaLJmzWqMMebYsWOme/fupkSJEsbb29tkz57dPPvss+bo0aOxtr//cTfY1qtXz9SrVy/W+zRv3jzzzjvvmPz58xsvLy/TsGFDc/DgwViv4cMPPzSFCxc23t7eplq1ambNmjWx9hmXEydOmEyZMpkmTZok2O+uu2H24MGDpnPnziYgIMBkyZLFvPjii+bGjRt2fb/88kvToEEDExgYaDw9PU3p0qXNRx99FGufhQoVMk8++aRZtmyZqVKlivHy8rIF68Tuwxhjli5daurWrWsyZ85s/P39TdWqVc2cOXOMMdHv7/3v/b0hMrG/H5JMz549zddff23KlCljMmXKZBYtWmRbFxISYusbFhZm+vbta/u9DAwMNI0bNzZbtmx5YE13f4ZnzJhhd/y9e/ea5557zuTMmdN4e3ubEiVKxAqDcSlYsKB58cUXY7XHFWaNMebZZ581kszp06dtbd9++62RZEaPHh3nMa5evWqyZs1qSpUqZWubO3eukWTefffdB9Z4144dO4wks3DhwkRvAyRV4r9HAZCgu18xZsuWzda2e/du1a5dW/nz59fgwYPl5+en+fPnq2XLlvr+++/VqlUrSdL169dVp04d7d27Vy+99JIqV66sixcvavHixTp58qRy5swpq9Wqp59+WuvWrdMrr7yi0qVLa+fOnZo8ebIOHDigH374Ic66qlatqiJFimj+/Pnq3Lmz3bp58+YpW7Zsatq0qaToUwH+7//+TxaLRb169VJgYKB++eUXde3aVWFhYXr99dfttn/77bfl6empAQMGKDw8PN6v13/66SdJUqdOneJcnylTJnXo0EGjRo3S+vXr1bhxY9u62bNn69q1a+rZs6du376tqVOnqmHDhtq5c6dy587t0Pt8V48ePRQYGKgRI0boxo0bkqRNmzbpzz//VLt27VSgQAEdO3ZMH3/8serXr689e/bI19dXdevWVZ8+ffT+++/rrbfeUunSpSXJ9m98xo0bJzc3Nw0YMEChoaEaP368OnbsqI0bN9r6fPzxx+rVq5fq1Kmjfv366dixY2rZsqWyZcumAgUKJLj/X375RXfu3NELL7yQYL/7tWnTRoULF9bYsWO1detWff7558qVK5fee+89u7oeeeQRPf3008qUKZN++ukn9ejRQ1arVT179rTb3/79+9W+fXu9+uqr6tatm0qWLOnQPmbOnKmXXnpJjzzyiIYMGaKsWbNq27ZtWrZsmTp06KChQ4cqNDRUJ0+e1OTJkyVJmTNnliSHfz9+++03zZ8/X7169VLOnDkVHBwc53v02muv6bvvvlOvXr1UpkwZXbp0SevWrdPevXtVuXLlBGuKyz///KM6derIw8NDr7zyioKDg3X48GH99NNPsU4HuNepU6d0/PhxVa5cOd4+97t7AVrWrFltbQ/6XQwICFCLFi00a9YsHTp0SMWKFdPixYslyaGfrzJlysjHx0fr16+P9fsHJDtnp2nA1dwdnVu5cqW5cOGCOXHihPnuu+9MYGCg8fLysn2Va4wxjRo1MuXKlbMbGbJaraZWrVqmePHitrYRI0bEO4px9yvFr776yri5ucX6mu+TTz4xksz69ettbfeOzBpjzJAhQ4yHh4e5fPmyrS08PNxkzZrVbrS0a9euJm/evObixYt2x2jXrp0JCAiwjZreHXEsUqRIor5KbtmypZEU78itMcYsXLjQSDLvv/++MSZmVMvHx8ecPHnS1m/jxo1GkunXr5+tLbHv893P7tFHH7X76tUYE+fruDuiPHv2bFtbQqcZxDcyW7p0aRMeHm5rnzp1qpFkG2EODw83OXLkMNWqVTORkZG2fjNnzjSSHjgy269fPyPJbNu2LcF+d90dmb1/pLxVq1YmR44cdm1xvS9NmzY1RYoUsWsrVKiQkWSWLVsWq39i9nH16lXj7+9vatSoEesr/3u/Vo/vK31Hfj8kGTc3N7N79+5Y+9F9I7MBAQGmZ8+esfrdK76a4hqZrVu3rvH39zf//vtvvK8xLitXroz1Lcpd9erVM6VKlTIXLlwwFy5cMPv27TMDBw40ksyTTz5p17dixYomICAgwWNNmjTJSDKLFy82xhhTqVKlB24TlxIlSpjHH3/c4e0ARzGbAZBEjRs3VmBgoIKCgvTss8/Kz89Pixcvto2iXb58Wb/99pvatGmja9eu6eLFi7p48aIuXbqkpk2b6uDBg7bZD77//ntVqFAhzhEMi8UiSVqwYIFKly6tUqVK2fZ18eJFNWzYUJL0+++/x1tr27ZtFRkZqYULF9rafv31V129elVt27aVFH2xyvfff6/mzZvLGGN3jKZNmyo0NFRbt26122/nzp3l4+PzwPfq2rVrkiR/f/94+9xdFxYWZtfesmVL5c+f37ZcvXp11ahRQ0uXLpXk2Pt8V7du3WJdkHPv64iMjNSlS5dUrFgxZc2aNdbrdlSXLl3sRq3r1KkjKfqiGknavHmzLl26pG7dutldeNSxY0e7kf743H3PEnp/4/Laa6/ZLdepU0eXLl2y+wzufV9CQ0N18eJF1atXT0eOHFFoaKjd9oULF7aN8t8rMftYsWKFrl27psGDB8e6gPLu70BCHP39qFevnsqUKfPA/WbNmlUbN260u1o/qS5cuKA1a9bopZdeUsGCBe3WPeg1Xrp0SZLi/XnYt2+fAgMDFRgYqFKlSmnChAl6+umnY00Ldu3atQf+nNz/uxgWFubwz9bdWpn+DamB0wyAJJo2bZpKlCih0NBQffnll1qzZo28vLxs6w8dOiRjjIYPH67hw4fHuY/z588rf/78Onz4sJ555pkEj3fw4EHt3btXgYGB8e4rPhUqVFCpUqU0b948de3aVVL0KQY5c+a0/c/+woULunr1qj777DN99tlniTpG4cKFE6z5rrv/I7x27ZrdV573ii/wFi9ePFbfEiVKaP78+ZIce58TqvvWrVsaO3asZsyYoVOnTtlNFXZ/aHPU/cHlbiC5cuWKJNnmDC1WrJhdv0yZMsX79fe9smTJIinmPUyOuu7uc/369QoJCdGGDRt08+ZNu/6hoaEKCAiwLcf385CYfRw+fFiSVLZsWYdew12O/n4k9md3/Pjx6ty5s4KCglSlShU98cQT6tSpk4oUKeJwjXf/eEnqa5QU7xR2wcHBmj59uqxWqw4fPqx3331XFy5ciPWHgb+//wMD5v2/i1myZLHV7mitiflDBHhYhFkgiapXr66qVatKih49fPTRR9WhQwft379fmTNnts3vOGDAgDhHq6TY4SUhVqtV5cqV06RJk+JcHxQUlOD2bdu21bvvvquLFy/K399fixcvVvv27W0jgXfrff7552OdW3tX+fLl7ZYTMyorRZ9T+sMPP+iff/5R3bp14+zzzz//SFKiRsvulZT3Oa66e/furRkzZuj1119XzZo1FRAQIIvFonbt2sU7V2dixTctU3zBxFGlSpWSJO3cuVMVK1ZM9HYPquvw4cNq1KiRSpUqpUmTJikoKEienp5aunSpJk+eHOt9iet9dXQfSeXo70dif3bbtGmjOnXqaNGiRfr11181YcIEvffee1q4cKEef/zxh647sXLkyCEp5g+g+/n5+dmda167dm1VrlxZb731lt5//31be+nSpbV9+3YdP3481h8zd93/u1iqVClt27ZNJ06ceOB/Z+515cqVOP8YBZIbYRZIBu7u7ho7dqwaNGigDz/8UIMHD7aN3Hh4eNj9TyYuRYsW1a5dux7YZ8eOHWrUqFGSRjvatm2rUaNG6fvvv1fu3LkVFhamdu3a2dYHBgbK399fUVFRD6zXUU899ZTGjh2r2bNnxxlmo6Ki9M033yhbtmyqXbu23bqDBw/G6n/gwAHbiKUj73NCvvvuO3Xu3FkTJ060td2+fVtXr16165cSI013J8A/dOiQGjRoYGu/c+eOjh07FuuPiPs9/vjjcnd319dff+3wRWAJ+emnnxQeHq7FixfbBZ+ETmlJ6j6KFi0qSdq1a1eCf+TF9/4/7O9HQvLmzasePXqoR48eOn/+vCpXrqx3333XFmYTe7y7P6sP+l2Py90/WI4ePZqo/uXLl9fzzz+vTz/9VAMGDLC990899ZS+/fZbzZ49W8OGDYu1XVhYmH788UeVKlXK9jk0b95c3377rb7++msNGTIkUce/c+eOTpw4oaeffjpR/YGHwTmzQDKpX7++qlevrilTpuj27dvKlSuX6tevr08//VRnzpyJ1f/ChQu2588884x27NihRYsWxep3d5SsTZs2OnXqlKZPnx6rz61bt2xX5cendOnSKleunObNm6d58+Ypb968dsHS3d1dzzzzjL7//vs4/2d7b72OqlWrlho3bqwZM2bEeYehoUOH6sCBA3rzzTdjjZj98MMPdue8/v3339q4caMtSDjyPifE3d091kjpBx98oKioKLs2Pz8/SYoVch9G1apVlSNHDk2fPl137tyxtc+ZMyfekbh7BQUFqVu3bvr111/1wQcfxFpvtVo1ceJEnTx50qG67o7c3n/KxYwZM5J9H4899pj8/f01duxY3b59227dvdv6+fnFedrHw/5+xCUqKirWsXLlyqV8+fIpPDz8gTXdLzAwUHXr1tWXX36p48eP26170Ch9/vz5FRQU5NCd3N58801FRkbajVY/++yzKlOmjMaNGxdrX1arVd27d9eVK1cUEhJit025cuX07rvvasOGDbGOc+3aNQ0dOtSubc+ePbp9+7Zq1aqV6HqBpGJkFkhGAwcO1HPPPaeZM2fqtdde07Rp0/Too4+qXLly6tatm4oUKaJz585pw4YNOnnypO12jwMHDrTdWeqll15SlSpVdPnyZS1evFiffPKJKlSooBdeeEHz58/Xa6+9pt9//121a9dWVFSU9u3bp/nz52v58uW20x7i07ZtW40YMULe3t7q2rWr3Nzs/54dN26cfv/9d9WoUUPdunVTmTJldPnyZW3dulUrV67U5cuXk/zezJ49W40aNVKLFi3UoUMH1alTR+Hh4Vq4cKFWr16ttm3bauDAgbG2K1asmB599FF1795d4eHhmjJlinLkyKE333zT1iex73NCnnrqKX311VcKCAhQmTJltGHDBq1cudL29e5dFStWlLu7u9577z2FhobKy8tLDRs2VK5cuZL83nh6emrkyJHq3bu3GjZs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Pqx58+apf//+CgkJ0datW1WhQgU1bdpU58+fj3ebhQsX6syZM7bHrl275O7urueee05S9LRaLVu21JEjR/Tjjz9q27ZtKlSokBo3bqwbN27Y9tOxY0etW7dOu3fvTvHXCQDIoEwGExoaaiSZ0NDQWOtu3bpl9uzZY27duuWEyh5O586dTYsWLezaWrdubSpVqmRbPn78uPHw8DD9+/ePtf37779vJJm//vrLGGPMxo0bjSQzZcqUOI935cqVeGs5ceKEadeuncmWLZvx9fU1VapUse03rjr79u1r6tWrZ1uuV6+e6dmzp+nbt6/JkSOHqV+/vmnfvr1p06aN3XYREREmR44cZtasWcYYY6KiosyYMWNMcHCw8fb2NuXLlzcLFiyIt05jjJkwYYKpWrWqXdvFixdNu3btTL58+YyPj48pW7as+eabb+z6xFWjMcbs3LnTNGvWzPj5+ZlcuXKZ559/3ly4cMG23S+//GJq165tAgICTPbs2c2TTz5pDh06lGCND6t69eqmZ8+etuWoqCiTL18+M3bs2ETvY/Lkycbf399cv37dGGPM/v37jSSza9cuu/0GBgaa6dOn223boEEDM2zYsDj368q/cwCAlJNQXrsfI7OJULWqVKBA6j+qVk16zbt27dKff/4pT09PW9t3332nyMjIWCOwkvTqq68qc+bM+vbbbyVJc+bMUebMmdWjR4849581a9Y4269fv6569erp1KlTWrx4sXbs2KE333xTVqvVofpnzZolT09PrV+/Xp988ok6duyon376SdevX7f1Wb58uW7evKlWrVpJksaOHavZs2frk08+0e7du9WvXz89//zz+uOPP+I9ztq1a1X1vjf69u3bqlKlipYsWaJdu3bplVde0QsvvBDrq/n7a7x69aoaNmyoSpUqafPmzVq2bJnOnTunNm3a2La5ceOG+vfvr82bN2vVqlVyc3NTq1atEnx/xowZo8yZMyf4OH78eJzbRkREaMuWLWrcuLGtzc3NTY0bN9aGDRviPeb9vvjiC7Vr105+fn6SpPDwcEmSt7e33X69vLy0bt06u22rV6+utWvXJvpYAAA4gpsmJMLZs9KpU86u4sF+/vlnZc6cWXfu3FF4eLjc3Nz04Ycf2tYfOHBAAQEByps3b6xtPT09VaRIER04cECSdPDgQRUpUkQeHh4O1fDNN9/owoUL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lKio6J/YM2ekvHmjLxN0d0+xwx05ckQ1a9ZUiRIl9O2336pw4cLavXu3Bg4cqF9++UV//fWXsmfP/lDHyJw5szJnzpxMFSfNgAED9Nprr9mWq1WrpldeeUXdunV76H1HRkbKw8PjofcDAMg4/v03Opzu35+4/pcvP7hPlizR+8uV6+Fqyyhc6qYJLmPhQik4WGrQQOrQIfrf4ODo9hTSs2dPeXp66tdff1W9evVUsGBBPf7441q5cqVOnTqloUOH2voGBwfr7bffVvv27eXn56f8+fNr2rRpduslqVWrVrJYLLblkSNHqmLFirZ+L774olq2bKkxY8Yod+7cypo1q0aPHq07d+5o4MCByp49uwoUKKAZM2bY1Tpo0CCVKFFCvr6+KlKkiIYPH67IyMhEvc7MmTMrT548toe7u7v8/f3t2u6yWq168803lT17duXJk0cjR46025fFYtHHH3+sp59+Wn5+fnr33XclST/++KMqV64sb29vFSlSRKNGjdKd/y4JNcZo5MiRKliwoLy8vJQvXz716dPHbr83b97USy+9JH9/fxUsWFCfffaZ3fqdO3eqYcOG8vHxUY4cOfTKK6/o+vXr8b7mGzduqFOnTsqcObPy5s2riRMnJuq9AgCkvK+/ljZsiA6piXncq29faf362I+TJwmyjiDMJreFC6O/Fzh50r791Kno9hQItJcvX9by5cvVo0ePWPPt5smTRx07dtS8efPsvn6fMGGCKlSooG3btmnw4MHq27evVqxYIUna9N+JPDNmzNCZM2dsy3H57bffdPr0aa1Zs0aTJk1SSEiInnrqKWXLlk0bN27Ua6+9pldffVUn73k//P39NXPmTO3Zs0dTp07V9OnTNXny5OR8SyRJs2bNkp+fnzZu3Kjx48dr9OjRttd418iRI9WqVSvt3LlTL730ktauXatOnTqpb9++2rNnjz799FPNnDnTFnS///57TZ48WZ9++qkOHjyoH374QeXuu23KxIkTVbVqVW3btk09evRQ9+7dtf+/P9lv3Lihpk2bKlu2bNq0aZMWLFiglStXqlevXvG+joEDB+qPP/7Qjz/+qF9//VWrV6/W1q1bk/ndAgAkxY0bMc/z54+epzUxj/btpTFjpFq1Yj/8/Z33elySyWBCQ0ONJBMaGhpr3a1bt8yePXvMrVu3krbzO3eMKVDAmOjzu2M/LBZjgoKi+yWjv/76y0gyixYtinP9pEmTjCRz7tw5Y4wxhQoVMs2aNbPr07ZtW/P444/bluPaX0hIiKlQoYJtuXPnzqZQoUImKirK1layZElTp04d2/KdO3eMn5+f+fbbb+Otf8KECaZKlSrxHichhQoVMpMnT47VXq9ePfPoo4/atVWrVs0MGjTItizJvP7663Z9GjVqZMaMGWPX9tVXX5m8efMaY4yZOHGiKVGihImIiIi3nueff962bLVaTa5cuczHH39sjDHms88+M9myZTPXr1+39VmyZIlxc3MzZ8+eNcZEv68tWrQwxhhz7do14+npaebPn2/rf+nSJePj42P69u0bZw2u5KF/5wDgIWzbZkznzsa0apX0R4kSMf+b/+03Z7+i9COhvHY/zplNTmvXxh6RvZcx0okT0f3q10/2wxsHLny6/0YTNWvWTNKV8Y888ojc3GIG+HPnzq2yZcvalt3d3ZUjRw6dP3/e1jZv3jy9//77Onz4sK5fv647d+4oSwrMCVK+fHm75bx589rVIUlVq1a1W96xY4fWr19vG4mVpKioKN2+fVs3b97Uc889pylTpqhIkSJq1qyZnnjiCTVv3lyZ7pn75N7jWiwW5cmTx3bcvXv3qkKFCvLz87P1qV27tqxWq/bv36/cuXPb1XP48GFFRESoRo0atrbs2bOrZMmSjr4dAJBu3LwpJfLstAS98IK0a9fD7+euB02DhZTBaQbJ6cyZ5O2XSMWKFZPFYtHevXvjXL93715ly5ZNgYGByXpcSbEumLJYLHG2Wf+ba2TDhg3q2LGjnnjiCf3888/atm2bhg4dqoh776mXgrVZ75vz5N5QKUXfyGPUqFHavn277bFz504dPHhQ3t7eCgoK0v79+/XRRx/Jx8dHPXr0UN26de3O+U3McQEASTNyZPQFUlmzPvwjOYNshQrSPeMOSEX8DZGc8uZN3n6JlCNHDjVp0kQfffSR+vXrZ3fe7NmzZzVnzhx16tRJFovF1v7XX3/Z7eOvv/5S6dKlbcseHh6KiopK1jol6c8//1ShQoXsLkj7999/k/04SVW5cmXt379fxYoVi7ePj4+PmjdvrubNm6tnz54qVaqUdu7cqcqVKz9w/6VLl9bMmTN148YNW5Bev3693Nzc4hxtLVq0qDw8PLRx40YVLFhQknTlyhUdOHBA9erVS+KrBIC07erV6EtM4rp5wANmZ0ySnDmlf/5J+vYWi5Q7d/S/SH2E2eRUp45UoED0xV5xfeVvsUSvr1Mn2Q/94YcfqlatWmratKneeecdu6m58ufPb/e1uRQdoMaPH6+WLVtqxYoVWrBggd1tgYODg7Vq1SrVrl1bXl5eypYtW7LUWbx4cR0/flxz585VtWrVtGTJEi1atChZ9p0cRowYoaeeekoFCxbUs88+Kzc3N+3YsUO7du3SO++8o5kzZyoqKko1atSQr6+vvv76a/n4+KhQoUKJ2n/Hjh0VEhKizp07a+TIkbpw4YJ69+6tF154IdYpBlL07A1du3bVwIEDlSNHDuXKlUtDhw61O7UDANKbzp2lxYsf3C85bgCaObPUr1+yjzMhFRFmk5O7uzR1avSsBRaLfaC9++falCkpMt9s8eLFtXnzZoWEhKhNmza6fPmy8uTJo5YtWyokJCTWHLNvvPGGNm/erFGjRilLliyaNGmS3W2BJ06cqP79+2v69OnKnz+/jj3MjaLv8fTTT6tfv37q1auXwsPD9eSTT2r48OGxps1ylqZNm+rnn3/W6NGj9d5778nDw0OlSpXSyy+/LEnKmjWrxo0bp/79+ysqKkrlypXTTz/9lOibUvj6+mr58uXq27evqlWrJl9fXz3zzDOaNGlSvNtMmDBB169fV/PmzeXv76833nhDoaGhyfJ6ASAlXL4s9eol7dmTtO137Hhwn5dflqZPT9r+kb5YjCNXDaUDYWFhCggIUGhoaKyLjm7fvq2jR4+qcOHC8vb2TvpBFi6Mnjzu3ovBgoKig2zr1knfbzJJjluuAskh2X7nAKQZkZFSz57JFzS//TZ2W9asUuPGXHCVniWU1+7Hj0FKaN1aatEiVe8ABgCAs4WHS2XKSEeO2Lcn5W/VzJmlsWOldu2SpzakX4TZlOLuniLTbwEAkFy2bZO++UZKrgllNm6MHWT37JHuub4YSHaE2Qwouc5/BQC4LqtVat48+prllDJ/PkEWKY8wCwBAOvHaa9K8edFB9UFu3UqeGw/EJVMmaflyqWHDlNk/cC/CbBwy2DVxgNPwuwbYs1qjL7e4dMnxbQ8flj79NGnHzZ8/+trl5FKoUPS8q0BqIMze4+6dm27evGl34wEAKePund/cuTgSkCQNHixNmJA8+0rs1/u5ckUfs1q15DkukNoIs/dwd3dX1qxZdf78eUnRc4JauJ0HkCKsVqsuXLggX19fZWJ+HWRgH38s/fJL9POffkqefb7/vtS7d/LsC0jr+D/IffLkySNJtkALIOW4ubmpYMGC/NGIDGvfPqlHj7jXjR+ftH0GB0stWya1IsD1EGbvY7FYlDdvXuXKlUuRKXVmPABJkqenJ7fmRYZ29mzsNoslelR14MDUrwdwRYTZeLi7u3MeHwDAIcZIa9ZImzcnrv+hQzHP+/SRhg2TPD2lgICUqQ9IjwizAAAkk02bkn6/HF9fKTAwWcsBMgS+3wMAIBlMmiTVqJH07R99NPlqATISRmYBAEgkq1XauTP2zQYuXZLeeMO+rXVrqX37xO23ZEmpXLnkqRHIaAizAAAkUuPG0u+/P7hf/frSzJmSv39KVwSAMAsAQDxu3ZKmT5cOHJBCQxMXZIcOld55J+VrAxCNMAsAQDy++krq2zfudb16xW4rUEB69dWUrQmAPcIsACBDO31aatVK2rUr9rqbN2O3WSzSp59K3bqlfG0AHowwCwDIcMLDpZUro8Pq1KnS338/eJsvv5TKl5dy5ZKCglK+RgCJQ5gFAGQ4rVtLS5fGve7+WQUsFunxx6UXX4x+DiBtIcwCANK1OXOkuXOlqKiYtl9+id3PYom+I1eRIqlXG4CHR5gFAKQ7t29LYWHS5cvS888n3HfixOh/69YlyAKuiDALAEhX1qyRWrSQrl5NuJ+XV3SQ7dkzVcoCkEIIswCAdOWbb+IOsi+/LI0fH7Ps5SX5+qZaWQBSCGEWAJCu3LkT87x+fSlLFqlgweibGWTL5rSyAKQQwiwAIN364AOpbFlnVwEgJRFmAQAuLypK2r5dioyUzp93djUAUhNhFgDg0oyR6tWT1q93diUAnIEwCwBwOTdvSh9/LB08GD39VlxB1ttbyp8/9WsDkLoIswAAlzNzpjRgQNzrXn9dcnOLnp6LC76A9I8wCwBwOUePxm5zc5O+/FLq3Dn16wHgPIRZAIBLmzFDqlRJCgyU8uVzdjUAUhthFgDg0ooVkypUcHYVAJyFMAsASNP27ZPeflu6cCGmbf9+59UDIG0hzAIA0rSRI6V58+Jf7+mZaqUASIPcnF0AAAAJOXcu/nV160pVqqReLQDSHkZmAQBOY4y0bJm0bVv8ff79N+b5uXPR88dKksUi+funbH0A0j7CLADAaf74Q3riicT3z5IlJswCgMRpBgAAJ9q9O/F9/+//CLIAYmNkFgCQLK5fl3bscGybw4djnvfrJzVoEHc/Ly+pXr2k1wYg/SLMAgAe2pUrUtGi0f8mVeXKUvPmyVcTgIyB0wwAAA9tw4aHC7KSVLx48tQCIGNhZBYA8NCMiXleq5ZUs6Zj29esKVWvnrw1AcgYCLMAgIcyfLj0zjsxy48/Lg0b5rx6AGQshFkAgMO2bJH27JEiI+2DrCTlyOGcmgBkTIRZAIBDli2LHn2NS+fOUvv2qVsPgIyNMAsAeKDly6UPPpBu3ZJ++y3uPv36SZMmpW5dAECYBQDEKSJCOns2+nmzZnH3eeYZqWFDKTBQevrp1KsNAO4izAIAYjl6NPqOW+fPx9+nZk3pm28kT8/UqwsA7sc8swCAWJYsiTvINmgg3bgR/fjzT4IsAOdjZBYAEEtUVMzz//s/qWBBKVu26PNifX2dVxcA3I8wCwAZ3J49Uvfu0vHjMW2hoTHP+/aV2rVL/boAIDEIswCQgV2+LLVpI+3eHX+fzJlTrx4AcBRhFgAyqH//lUqVkm7ftm/PlSvmee3aUpMmqVsXADiCMAsAGUBYmDR1qnT4cEzbrFn2ffz9pdOnGYkF4FoIswCQAXz2mTRiRPzrAwOjZzAgyAJwNYRZAEjHwsOlxx6T1qyJv0/27NLmzdEzFgCAqyHMAkA6tnp17CC7ZIlUpEjMcqFCko9PqpYFAMmGMAsA6dj9F3eNHy898YRzagGAlMAdwAAgnVqwQGrZMmZ5zBhp4ECnlQMAKYKRWQBwUVevSleuxL0uMjJ6/th7eXmleEkAkOoIswDggr75RnrxxejQmhglSkjPPJOiJQGAUxBmASANOHdO+uqr6PlgE+PttxO/7/btpTlzJIslabUBQFpGmAWANOCVV6TFi5O2bbNmUrZsca/LlUt64w2CLID0izALAE506JDUtWvC88AmpGpVaelSwiqAjIswCwBONH167CD766+J29bDQ6pViyALIGMjzAKAE127Zr/81VdSkybOqQUAXBFhFgCSQXi4NHWqtHOnY9tt3BjzfOtWqVKl5K0LANI7wiwAJIOFC6VBgx5uH+7uyVMLAGQk3AEMAJLBiRMPt33lytIjjyRPLQCQkTAyCwDJbNo06bHHEt/fzU0qXJgLuQAgKQizAJDM8uaVihVzdhUAkDFwmgEAAABcFmEWAAAALoswCwAAAJdFmAUAAIDLIswCAADAZTGbAQA4aNs26ZVXpFOnYtquX3dePQCQkRFmASABVqv0999SWFhM20sv2QfZ+/n7p3xdAIBohFkASED37tJnn8W/3mKRgoJiluvWlerXT/GyAAD/IcwCQAJWrIh/XY4c0smTkrd36tUDALBHmAWAe0RGRp9acJcx0f/6+koDBsS0Z8oktWpFkAUAZyPMAsB/evSQPv3UPszelTmzNGpU6tcEAEgYYRZAhmWM9Ntv0sGD0Rd4ffxx/H0DA1OvLgBA4hFmAWRY8+dL7drFva5evZjnmTPbn2IAAEg7CLMAMpQFC6QPP5TCw6WNG+PuM2qUNGJE6tYFAEgawiyAdCsqKvoUgrsXcRkjtWkTd9+ePaVq1aQ8eaTGjVOvRgDAwyHMAkiXIiOlihWlPXsS7mexSE8+Kb3/vuTGDb4BwOXwn24A6dKOHQkH2XbtokdqrVbpp58IsgDgqhiZBZAu3Tu9VunS0v/9X8xyzpxS796pXxMAIPkRZgGkO198Ib38csxy48bRpxEAANIfp3+xNm3aNAUHB8vb21s1atTQ33//nWD/KVOmqGTJkvLx8VFQUJD69eun27dvp1K1AFzB8OH2y1myOKcOAEDKc2qYnTdvnvr376+QkBBt3bpVFSpUUNOmTXX+/Pk4+3/zzTcaPHiwQkJCtHfvXn3xxReaN2+e3nrrrVSuHEBaduNGzPPHHpO6dXNeLQCAlOXUMDtp0iR169ZNXbp0UZkyZfTJJ5/I19dXX375ZZz9//zzT9WuXVsdOnRQcHCwHnvsMbVv3/6Bo7kAMobTp6W+faPv5iVFnyu7fLlUqJBz6wIApBynhdmIiAht2bJFje+Z0NHNzU2NGzfWhg0b4tymVq1a2rJliy28HjlyREuXLtUTTzwR73HCw8MVFhZm9wCQ/hgTfXrBvefGZuKqAABI95z2n/qLFy8qKipKuXPntmvPnTu39u3bF+c2HTp00MWLF/Xoo4/KGKM7d+7otddeS/A0g7Fjx2rUqFHJWjuAtOXcuejbz+7fb9/epYtz6gEApB6XGrdYvXq1xowZo48++kg1atTQoUOH1LdvX7399tsafv8VH/8ZMmSI+vfvb1sOCwtTUFBQapUMIJnduSP98ov0778xbV98ETvIHj8u8asOAOmf08Jszpw55e7urnPnztm1nzt3Tnny5Ilzm+HDh+uFF17Qy//NuVOuXDnduHFDr7zyioYOHSq3OGY99/LykpeXV/K/AABO8cknD54jds4cgiwAZBROO2fW09NTVapU0apVq2xtVqtVq1atUs2aNePc5ubNm7ECq7u7uyTJ3L35OoB06c6d6BD7oCC7b5/UoUPq1AQAcD6nnmbQv39/de7cWVWrVlX16tU1ZcoU3bhxQ13+O9GtU6dOyp8/v8aOHStJat68uSZNmqRKlSrZTjMYPny4mjdvbgu1ANKnVaukDz+0bxs9WipcOGa5enWpRInUrQsA4FxODbNt27bVhQsXNGLECJ09e1YVK1bUsmXLbBeFHT9+3G4kdtiwYbJYLBo2bJhOnTqlwMBANW/eXO+++66zXgKAVHL5sv3yiy/GvjkCACDjsZgM9v18WFiYAgICFBoaqizcFghIk27flmbMkA4dimnbt09aujT6+dSpUp8+zqkNAJDyHMlrLjWbAYCMYcYMqUeP+NdbLKlXCwAgbSPMAnCqlSulXr2kCxdi2u4/peBe3t7SPfdaAQBkcIRZAE5z8KDUpEnCfT77THrkkZjlkiWlHDlSti4AgOsgzAJwijVrou/ada9ChSRPz+jnFovUvLn08sucVgAAiB9hFkCKO3FCGjdOOnMmpm3RIvs+JUpEX+RFcAUAOIIwC+ChGSOFhcW/ftAg6dtv41/ftKn06acEWQCA4wizAB7K1atSzZrRo6pJUaZM9Citj0+ylgUAyCAIswAeyrJljgXZQ4ckX9+Y5Tx5GJEFACQdYRbAQ4mMjHleurRUsGDc/TJlir5rV9GiqVIWACCDIMwCSDa9eiV8swMAAJKbm7MLAAAAAJKKMAsAAACXxWkGABx24oT0+efRMxns3evsagAAGRlhFoDDevaUfvopdjuzEgAAUhthFsADRUZKzz4rrV4dvRzXDRJ8faUmTVK1LAAACLMAHmzNGmnx4rjXbdwY/W+JElLWrKlWEgAAkgizABLh5s2Y5zlzSoGB0SOxgwZJ1as7ry4AAAizABzSr5/01lvOrgIAgGiEWQA2xkhXrkT/e6+4zpEFACAtIMwCkCTduSPVqSP99ZezKwEAIPG4aQIAnT0rvfRS4oJsnjwpXw8AAInFyCwANW8ubd5s3/bEE7H7lS0rtW2bOjUBAJAYhFkA+ucf++XFi6MDLgAAaR1hFkhPoqKktWulM2ekvHmjT4J1d7frEhkp7dwpWa0xbfc+37FDKl8+leoFAOAhEWaB9GLhQqlvX+nkyZi2AgWkqVOl1q0lRWfd8uWlffvi3kWlSgRZAIBr4QIwID1YuDD6frP3BllJOnUqun3hQh04ILVoEX+QlaSiRVO2TAAAkhsjs4Cri4qKHpG9f3JYKbrNYpFef12d87XSXxstdqt79Yp5nj279MorKVwrAADJjDALuLq1a20jst+qnd7QRIUqIGa9kXRCunkiJshmyiT98IP05JOpWyoAAMmNMAu4ujNnbE/H602dUb4HbnLiBPPFAgDSB8Is4Ory5rU9vSUfSZKbovSIdtv3K1JEWfJm1ogRBFkAQPpBmAVcXZ060bMWnDoVfUqBpCwK0z+qEL1gsUSvP3BUco9/NwAAuCJmMwBcnbt79PRbcbH8d57slCmx5psFACA9IMwC6UHr1tJ330Vf2XWvAgWi2/+bZxYAgPSG0wyA9KJ1a6mokfZL8vWTlvwe5x3AAABITwizQLry32kFnp5S/fpOrQQAgNRAmAVcWFSU1KOHtGZN9PKRI86tBwCA1EaYBVzYmjXSZ5/Fbvf1Tf1aAABwBsIs4MJCQ2Oee3lJ3t6Sn58UEuK8mgAASE2EWSCdGDVKGjTI2VUAAJC6mJoLAAAALoswC7iYH3+U8uWLPq3g2WedXQ0AAM7FaQaAi5kyRTpzJnZ79uypXgoAAE5HmAVczK1bMc8rVoz+t1w5qU0bp5QDAIBTEWYBF7Ztm7MrAADAuThnFgAAAC6LMAsAAACXxWkGQBp24oS0YIF0+3ZM26lTzqsHAIC0hjALpGFPPSX984+zqwAAIO3iNAMgDdq0SXr00YSDbIMGqVcPAABpFSOzQBo0YYK0fr192+LFMc+9vaV69VK3JgAA0iLCLJAGhYXFPM+dW5oxQ3r8cefVAwBAWkWYBdK4/fulgABnVwEAQNrEObMAAABwWYRZIA05c0aqXFlavtzZlQAA4BoIs0AaceaM1Lq1/S1qfX0lLy/n1QQAQFrHObNAGnDzplSmjHT1qn37Rx9Fz1wAAADiRpgFnGz1aqlTJ/sga7FIBw9KRYs6qyoAAFwDYRZwomvXou/ydeNGTFumTNKWLQRZAAASg3NmASf5+efoOWTvDbJ580p//imVL++8ugAAcCWMzAJO8u230q1bMcvVq0cHWXd359UEAICrYWQWcJKoqJjnjz0mffEFQRYAAEcxMguksps3pRdflBYsiGn77DOpUCGnlQQAgMtiZBZIZT//bB9kpej5ZAEAgOMIs0AqCwuzXx46VAoMdE4tAAC4Ok4zAFLRr79K3brFLH/+udS1q/PqAQDA1TEyC6SSa9ekVq3s29z4DQQA4KHwv1IglVy6FH3x1105c0pNmzqvHgAA0gPCLOAEFStKJ05I+fI5uxIAAFwbYRZwglKlJG9vZ1cBAIDrI8wCqeDPP6XmzZ1dBQAA6Q+zGQAp5M6d6FMJJKl1a+ncuZh1Xl7OqQkAgPSGMAukgKtXpQoVpOPHY6/LnVvq0iXVSwIAIF0izALJbO9e6fnn4w6yjzwi/fMPU3IBAJBcCLNAMrpzR2rSRDp1yr69fXvJz0/q0YMgCwBAciLMAsno+nX7IOvuLm3aJFWq5LyaAABIzwizQApxd5f+/VfKn9/ZlQAAkH7xhSeQQpo0IcgCAJDSCLNAMjl8WOrb19lVAACQsXCaAeAgqzX6cb/XX5d+/jlm2d091UoCACDDIswCDvjpp+g5Yi9dSrifh4fUrl3q1AQAQEZGmAUSwRhp40bp6acT1//sWSl79pStCQAAEGaBRFm1KvqCrntVqhQ9d+y9vL2lPn0IsgAApBbCLPAAEydKAwbYt5UpI23ZIlkszqkJAABEYzYDIAHr1sUOsh07SmvWEGQBAEgLCLNAPPr2lerUsW+rU0f64gspRw7n1AQAAOwRZoF4fP21/fK4cdEjsl5ezqkHAADERpgF4hEVFfN85Eipe3enlQIAAOLBBWDAA5QqJYWEOLsKAAAQF0ZmAQAA4LIIswAAAHBZhFkAAAC4LMIsAAAAXBZhFgAAAC6LMAsAAACXRZgFAACAyyLMAvc5eFBq3VoKDXV2JQAA4EG4aQJwj1u3pK5dpbVrY9q8vZ1XDwAASBhhFvjP8eNSxYrSlSv27f36OaUcAACQCIRZ4D/LltkHWX9/6exZydfXeTUBAICEcc4s8B+rNea5p6f0ww8EWQAA0rqHCrO3b99OrjqANOWzz6SGDZ1dBQAAeBCHw6zVatXbb7+t/PnzK3PmzDpy5Igkafjw4friiy8cLmDatGkKDg6Wt7e3atSoob///jvB/levXlXPnj2VN29eeXl5qUSJElq6dKnDx0XGYYy0YYO0dGnCj127nF0pAABwlMPnzL7zzjuaNWuWxo8fr27dutnay5YtqylTpqhr166J3te8efPUv39/ffLJJ6pRo4amTJmipk2bav/+/cqVK1es/hEREWrSpIly5cql7777Tvnz59e///6rrFmzOvoykIEMGCBNmuTsKgAAQEqwGGOMIxsUK1ZMn376qRo1aiR/f3/t2LFDRYoU0b59+1SzZk1duf9S8ATUqFFD1apV04cffigpetQ3KChIvXv31uDBg2P1/+STTzRhwgTt27dPHh4ejpRtExYWpoCAAIWGhipLlixJ2gfSrtWrpc8/l8LDY9q++87x/WzaJFWtmmxlAQAABziS1xwemT116pSKFSsWq91qtSoyMjLR+4mIiNCWLVs0ZMgQW5ubm5saN26sDRs2xLnN4sWLVbNmTfXs2VM//vijAgMD1aFDBw0aNEju7u5xbhMeHq7we5JNWFhYomuEazFGat8+egaC+Iwe/eD91KhBkAUAwFU4HGbLlCmjtWvXqlChQnbt3333nSpVqpTo/Vy8eFFRUVHKnTu3XXvu3Lm1b9++OLc5cuSIfvvtN3Xs2FFLly7VoUOH1KNHD0VGRiokJCTObcaOHatRo0Ylui64Lqs14SDbt680fHjq1QMAAFKew2F2xIgR6ty5s06dOiWr1aqFCxdq//79mj17tn7++eeUqNHGarUqV65c+uyzz+Tu7q4qVaro1KlTmjBhQrxhdsiQIerfv79tOSwsTEFBQSlaJ5yvalVp0aKYZW9vKWdO59UDAABShsNhtkWLFvrpp580evRo+fn5acSIEapcubJ++uknNWnSJNH7yZkzp9zd3XXu3Dm79nPnzilPnjxxbpM3b155eHjYnVJQunRpnT17VhEREfL09Iy1jZeXl7y8vBJdF9IHLy+pQAFnVwEAAFJakuaZrVOnjlasWKHz58/r5s2bWrdunR577DGH9uHp6akqVapo1apVtjar1apVq1apZs2acW5Tu3ZtHTp0SNZ7Zrc/cOCA8ubNG2eQBQAAQPrmcJgtUqSILl26FKv96tWrKlKkiEP76t+/v6ZPn65Zs2Zp79696t69u27cuKEuXbpIkjp16mR3gVj37t11+fJl9e3bVwcOHNCSJUs0ZswY9ezZ09GXgXTo9GlnVwAAAFKbw6cZHDt2TFFRUbHaw8PDderUKYf21bZtW124cEEjRozQ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7N2717z44osmICDAnDx5MpUrx12OfoZ3HT161OTPn9/UqVPHtGjRInWKRZwc/QzDw8NN1apVzRNPPGHWrVtnjh49alavXm22b9+eypXDGMc/vzlz5hgvLy8zZ84cc/ToUbN8+XKTN29e069fv1SuHHctXbrUDB061CxcuNBIMosWLUqw/5EjR4yvr6/p37+/2bNnj/nggw+Mu7u7WbZsWeoUfA/C7EOqXr266dmzp205KirK5MuXz4wdOzbO/m3atDFPPvmkXVuNGjXMq6++mqJ1In6Ofob3u3PnjvH39zezZs1KqRLxAEn5DO/cuWNq1aplPv/8c9O5c2fCrJM5+hl+/PHHpkiRIiYiIiK1SkQCHP38evbsaRo2bGjX1r9/f1O7du0UrROJk5gw++abb5pHHnnErq1t27amadOmKVhZ3DjN4CFERERoy5Ytaty4sa3Nzc1NjRs31oYNG+LcZsOGDXb9Jalp06bx9kfKSspneL+bN28qMjJS2bNnT6kykYCkfoajR49Wrly51LVr19QoEwlIyme4ePFi1axZUz179lTu3LlVtmxZjRkzRlFRUalVNv6TlM+vVq1a2rJli+1UhCNHjmjp0qV64oknUqVmPLy0lGcypfoR05GLFy8qKipKuXPntmvPnTu39u3bF+c2Z8+ejbP/2bNnU6xOxC8pn+H9Bg0apHz58sX6pUbqSMpnuG7dOn3xxRfavn17KlSIB0nKZ3jkyBH99ttv6tixo5YuXapDhw6pR48eioyMVEhISGqUjf8k5fPr0KGDLl68qEcffVTGGN25c0evvfaa3nrrrdQoGckgvjwTFhamW7duycfHJ9VqYWQWeAjjxo3T3LlztWjRInl7ezu7HCTCtWvX9MILL2j69OnKmTOns8tBElmtVuXKlUufffaZqlSporZt22ro0KH65JNPnF0aEmH16tUaM2aMPvroI23dulULFy7UkiVL9Pbbbzu7NLggRmYfQs6cOeXu7q5z587ZtZ87d0558uSJc5s8efI41B8pKymf4V3/+9//NG7cOK1cuVLly5dPyTKRAEc/w8OHD+vYsWNq3ry5rc1qtUqSMmXKpP3796to0aIpWzTsJOX3MG/evPLw8JC7u7utrXTp0jp79qwiIiLk6emZojUjRlI+v+HDh+uFF17Qyy+/LEkqV66cbty4oVdeeUVDhw6Vmxtj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jlqyZIm+/PJL24Vfd40YMUKzZ8/WqFGjtHv3bu3du1dz587VsGHDHHptNWvWVPny5TVmzBhJUvHixbV582YtX75cBw4c0PDhw7Vp0ya7bYKDg/XPP/9o//79unjxoiIjI9WxY0flzJlTLVq00Nq1a3X06FGtXr1affr00cmTJx2qCUD6QZgFkO5VqFBBkyZN0nvvvaeyZctqzpw5dtNa3c/d3V2XLl1Sp06dVKJECbVp00aPP/64Ro0aJUkqX768/vjjDx04cEB16tRRpUqVNGLECOXLly/JNfbp00f9+/fXG2+8oXLlymnZsmVavHixihcvLin6FIXx48eratWqqlatmo4dO6alS5faRprvlSlTJr3//vv69NNPlS9fPrVo0SLe4zZs2FDZs2fX/v371aFDB7t1TZs21c8//6xff/1V1apV0//93/9p8uTJKlSokMOvr1+/fvr888914sQJvfrqq2rdurXatm2rGjVq6NKlS3ajtJLUrVs3lSxZUlWrVlVgYKDWr18vX19frVmzRgULFlTr1q1VunRpde3aVbdv32akFsjALMYY4+wiAAAAgKRgZBYAAAAuizALAAAAl0WYBQAAgMsizAIAAMBlEWYBAADgsgizAAAAcFmEWQAAALgswiwAAABcFmEWAAAALoswCwAAAJdFmAUAAIDL+n9d3A3ZqYb+YwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 800x800 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import roc_curve, auc\n",
    "\n",
    "# Assuming you have already computed y_pred_proba using your CRNN model\n",
    "y_pred_proba = crnn_model.predict(X_test)\n",
    "\n",
    "# Compute ROC curve and ROC area for each class\n",
    "fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba[:, 1])  # Assuming y_test is binary (0 or 1)\n",
    "\n",
    "roc_auc = auc(fpr, tpr)\n",
    "\n",
    "# Plot ROC curve\n",
    "plt.figure(figsize=(8, 8))\n",
    "plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)\n",
    "plt.title('Receiver Operating Characteristic (ROC)')\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.legend(loc='lower right')\n",
    "\n",
    "# Mark optimal threshold point on ROC curve\n",
    "optimal_threshold_index = np.argmax(tpr - fpr)\n",
    "optimal_threshold = thresholds_roc[optimal_threshold_index]\n",
    "plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], c='red', marker='o', label='Optimal Threshold')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5187c6d5-f6ba-4133-bc7b-9b8ac9320a80",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "274ff05a-c233-4869-bd9d-c62d9b318753",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88c95ccc-51e7-4557-afaa-276a326f0e6d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9bb1fb17-964f-46d5-b179-6dfe1bd5b32d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
